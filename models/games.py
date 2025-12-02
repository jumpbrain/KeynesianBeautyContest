import os
import streamlit as st
import pymongo
from typing import Any, Dict, List, Optional, Self
from datetime import datetime
from pydantic import BaseModel, model_validator
import pandas as pd
import trueskill
from trueskill import Rating
import json
from pathlib import Path


# CSV fallback file for environments without MongoDB
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "games.csv"


def _ensure_csv_dir(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


class Result(BaseModel):
    """
    An individual result capturing the cumulative score for an LLM in a game.
    and its rank, where 0 means that it won
    """

    name: str
    llm: str
    score: float
    rank: int

    @model_validator(mode="before")
    @classmethod
    def _backward_compat(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Allow legacy payloads that stored 'coins'
        if "score" not in values and "coins" in values:
            values["score"] = values.pop("coins")
        return values

    def __init__(self, **args):
        """
        If necessary, fix this up so that different minor model variants have the same name
        :return: self
        """
        super().__init__(**args)
        if self.llm.startswith("claude-3-5-sonnet"):
            self.llm = "claude-3.5-sonnet"

    def __repr__(self) -> str:
        """
        Convert this object to json
        :return: a json string for each result
        """
        return json.dumps(self.model_dump())

    def update_on(self, df: pd.DataFrame) -> None:
        """
        Add this result to the DataFrame for the leaderboard
        This is perhaps more complex than needed because the dataframe only has Win %, not a count of Wins
        :param df: the dataframe for the leaderboard
        """
        llm = self.llm
        if not (df["LLM"] == llm).any():
            df.loc[len(df)] = [llm, 0, 0.0, 0.0]
        games = df.loc[df["LLM"] == llm, "Games"].values[0]
        wins_percent = df.loc[df["LLM"] == llm, "Win %"].values[0]
        wins = games * wins_percent / 100
        df.loc[df["LLM"] == llm, "Games"] += 1
        if self.rank == 0:
            df.loc[df["LLM"] == llm, "Win %"] = (wins + 1) * 100 / (games + 1)
        else:
            df.loc[df["LLM"] == llm, "Win %"] = wins * 100 / (games + 1)


class Game(BaseModel):
    """
    The result of a Game, stored in the DB
    """

    run_date: datetime
    results: List[Result]

    def __str__(self) -> str:
        """
        :return: a string json version of this game
        """
        return json.dumps(self.model_dump())

    @staticmethod
    @st.cache_resource
    def get_connection():
        mongo_uri = os.getenv("MONGO_URI")
        return pymongo.MongoClient(mongo_uri)

    @classmethod
    @st.cache_data(ttl=1)
    def all(cls) -> List[Self]:
        client = cls.get_connection()
        games = client.beauty.games
        docs = list(games.find())
        return [Game(**doc) for doc in docs]

    def save(self):
        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri:
            client = self.get_connection()
            games = client.beauty.games
            games.insert_one(self.model_dump())
            return

        # CSV fallback: append this game to a CSV file with columns run_date, results_json
        path = Path(os.getenv("GAMES_CSV", DEFAULT_CSV))
        _ensure_csv_dir(path)
        row = {
            "run_date": self.run_date.isoformat(),
            "results": json.dumps([r.model_dump() for r in self.results]),
        }
        write_header = not path.exists()
        with path.open("a", encoding="utf-8") as fh:
            if write_header:
                fh.write("run_date,results\n")
            # Escape quotes for CSV; results is JSON string
            results_escaped = row["results"].replace('"', '""')
            fh.write(f'{row["run_date"]},"{results_escaped}"\n')

    @classmethod
    @st.cache_data(ttl=1)
    def count(cls) -> int:
        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri:
            client = cls.get_connection()
            return client.beauty.games.count_documents({})
        path = Path(os.getenv("GAMES_CSV", DEFAULT_CSV))
        if not path.exists():
            return 0
        # Count non-empty data rows (exclude header)
        with path.open("r", encoding="utf-8") as fh:
            lines = [l for l in fh.readlines() if l.strip()]
        # subtract header if present
        if lines and lines[0].startswith("run_date"):
            return max(0, len(lines) - 1)
        return len(lines)

    @classmethod
    @st.cache_data(ttl=2)
    def reset(cls):
        client = cls.get_connection()
        client.beauty.games.delete_many({})

    @classmethod
    @st.cache_data(ttl=1)
    def latest(cls, k: int) -> List[Self]:
        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri:
            client = cls.get_connection()
            games = client.beauty.games
            latest = games.find().sort({"run_date": -1}).limit(k)
            return [Game(**each) for each in latest]

        path = Path(os.getenv("GAMES_CSV", DEFAULT_CSV))
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i == 0 and line.startswith("run_date"):
                    continue
                if not line.strip():
                    continue
                # split on first comma, rest is JSON
                parts = line.strip().split(",", 1)
                if len(parts) != 2:
                    continue
                run_date_str, results_json = parts
                # strip enclosing quotes if present
                if results_json.startswith('"') and results_json.endswith('"'):
                    results_json = results_json[1:-1].replace('""', '"')
                try:
                    results_list = json.loads(results_json)
                    results = [Result(**r) for r in results_list]
                    g = Game(run_date=datetime.fromisoformat(run_date_str), results=results)
                    rows.append(g)
                except Exception:
                    continue
        # newest last row is the newest game written; reverse to get descending
        rows = list(reversed(rows))
        return rows[:k]

    @classmethod
    def ratings_for(cls, games: List[Self], df: pd.DataFrame) -> Dict[str, Rating]:
        """
        Create a Rating object for each LLM in the list of historic games
        :param games: a list of historic games
        :param df: a dataframe with the ranks of past games
        :return: a dictionary that maps models to their Rating objects
        """
        ratings = {row["LLM"]: Rating() for _, row in df.iterrows()}
        for game in games:
            llms = [result.llm for result in game.results]
            rating_groups = [(ratings[llm],) for llm in llms]
            ranks = [result.rank for result in game.results]
            rated = trueskill.rate(rating_groups, ranks=ranks)
            for index, llm in enumerate(llms):
                ratings[llm] = rated[index][0]
        return ratings

    @classmethod
    def games_df(cls) -> pd.DataFrame:
        """
        Create a dataframe that represents the leaderboard for games played
        Use the TrueSkill methodology to assess Skill level
        The expose method calculates 1 number (mean - 3 * standard deviation) for the leaderboard
        :return: a DataFrame with the leaderboard including Win % and Skill
        """
        columns = ["LLM", "Games", "Win %", "Skill"]
        df = pd.DataFrame(columns=columns)
        games = cls.all()
        for game in games:
            for result in game.results:
                result.update_on(df)
        ratings = cls.ratings_for(games, df)
        for llm, rating in ratings.items():
            df.loc[df["LLM"] == llm, "Skill"] = trueskill.expose(rating)
        return df

    @classmethod
    def latest_df(cls) -> pd.DataFrame:
        """
        Create a table of the most recent 5 games
        :return: A dataframe to represent the winners of the last 5 games
        """
        columns = ["When", "Winner(s)"]
        df = pd.DataFrame(columns=columns)
        for game in cls.latest(5):
            when = game.run_date
            winners = [result.llm for result in game.results if result.rank == 0]
            winners_str = ", ".join(winners)
            df.loc[len(df)] = [when, winners_str]
        return df

    @classmethod
    def csv_path(cls) -> Path:
        """Return the configured CSV path for games."""
        return Path(os.getenv("GAMES_CSV", DEFAULT_CSV))
