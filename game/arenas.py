import os
import logging
from typing import List, Self, Callable
from game.players import Player
from game.referees import Referee
import random
import pandas as pd
import math
from scipy.stats import rankdata
from models.games import Result, Game
from datetime import datetime
from interfaces.llms import LLM

ProgressCallback = Callable[[float, str], None]


class Arena:
    """
    Central manager for the Beauty LLM arena, maintaining the active players
    """

    players: List[Player]
    turn: int
    is_game_over: bool

    NAMES = ["Alex", "Blake", "Charlie", "Drew", "Eden", "Fallon", "Gale", "Harper"]
    TEMPERATURE = 0.7

    def __init__(self, players: List[Player]):
        """
        Create a new instance of the Arena, the manager of the game
        Set the 'other players' field for each player. Shuffle it to reduce any bias on the order in which players
        are listed.
        :param players: the players to use
        """
        self.players = players
        self.turn = 1
        self.is_game_over = False
        from datetime import datetime
        # Timestamp for this run; used to associate per-turn logs
        self.run_date = datetime.now().isoformat()
        self.apply_starter_policy(shuffle_opponents=True)

    def _assign_opponents(self, shuffle: bool = True) -> None:
        """Assign each player's list of opponents, optionally shuffled."""
        for player in self.players:
            others = [p for p in self.players if p is not player]
            if shuffle:
                random.shuffle(others)
            player.others = others

    def _determine_starter(self):
        """Resolve the starter player based on UI settings."""
        starter_name = "Vanilla"
        randomize = False
        manual_enabled = False
        try:
            import streamlit as st

            randomize = bool(st.session_state.get("randomize_starter", False))
            manual_enabled = bool(st.session_state.get("manual_starter_enabled", False))
            starter_name = st.session_state.get("starter_player", starter_name)
        except Exception:
            pass

        if not self.players:
            return None, randomize, manual_enabled

        if randomize:
            return random.choice(self.players), True, manual_enabled

        if manual_enabled:
            for player in self.players:
                if player.name == starter_name:
                    return player, False, True

        for player in self.players:
            if player.name == "Vanilla":
                return player, False, manual_enabled
        return self.players[0], False, manual_enabled

    def apply_starter_policy(self, shuffle_opponents: bool = False) -> None:
        """Reorder players so the selected starter acts first and sync UI state."""
        starter, randomize, manual_enabled = self._determine_starter()
        if starter is None:
            return

        if self.players and self.players[0] is not starter:
            ordered = [starter] + [p for p in self.players if p is not starter]
            self.players = ordered

        shuffle_flag = shuffle_opponents or randomize
        self._assign_opponents(shuffle=shuffle_flag)

        try:
            import streamlit as st

            st.session_state["starter_player_active"] = starter.name
            if randomize or not manual_enabled:
                st.session_state["starter_player"] = starter.name
        except Exception:
            pass

    def __repr__(self) -> str:
        """
        :return: a string to represent the arena
        """
        result = f"Arena at turn {self.turn} with {len(self.players)} players:\n"
        for player in self.players:
            result += f"{player}\n"
        return result

    def do_save_game(self, names: List[str], llms: List[str], scores: List[float], ranks: List[int]):
        results = []
        for name, llm, score, rank in zip(names, llms, scores, ranks):
            r = Result(name=name, llm=llm, score=score, rank=rank)
            results.append(r)
        game = Game(run_date=datetime.now(), results=results)
        game.save()

    def save_game(self):
        if os.getenv("MONGO_URI"):
            try:
                names = [player.name for player in self.players]
                llms = [player.llm.model_name for player in self.players]
                scores = [player.score for player in self.players]
                ranks = rankdata([-score for score in scores], method="min") - 1
                ranks = list(ranks.astype(int))
                self.do_save_game(names, llms, scores, ranks)
            except Exception as e:
                logging.error("Failed to save game results")
                logging.error(e)

    def handle_game_over(self):
        """The game has ended - figure out who's a winner; there could be multiple"""
        self.is_game_over = True
        winning_score = max(player.score for player in self.players)
        for player in self.players:
            if player.score == winning_score:
                player.is_winner = True
        self.save_game()

    def prepare_for_turn(self) -> None:
        """Before carrying out a turn, store each player's score prior to the round."""
        self.apply_starter_policy()
        for player in self.players:
            player.prior_score = player.score

    def process_turn_outcome(self) -> None:
        """
        A turn has completed. Handle the outcome, including checking if the game has ended
        """
        for player in self.players:
            player.series.append(player.score)
        if self.turn == 10:
            self.handle_game_over()
        elif not self.is_game_over:
            self.turn += 1

    def do_turn(self, progress: ProgressCallback) -> bool:
        """
        Carry out a Turn by delegating to a Referee object
        :param progress: a callback on which to report progress
        :return True if the game ended
        """
        self.prepare_for_turn()
        ref = Referee(self.players, self.turn, run_date=self.run_date)
        ref.do_turn(progress)
        self.process_turn_outcome()
        return self.is_game_over

    @classmethod
    def model_names(cls) -> List[str]:
        """
        Determine the list of model names to use in a new Arena
        If there's an environment variable ARENA=random then pick 4 random model names
        otherwise use 4 cheap models
        The arena should support 3 or more names, although only 4 has been tested
        :return: a list of names of LLMs for a new Arena
        """
        # Default lineup: Vanilla baseline, Strategic reasoner, and Agressor red-team.
        # All three use the same underlying model by default for cost stability.
        return [
            "gpt-5-mini",
            "gpt-5-mini",
            "gpt-5-mini",
        ]

    @classmethod
    def default(cls) -> Self:
        """
        Return a new instance of Arena with default players
        :return: an Arena instance
        """
        # Use three players by default: Vanilla (baseline), Strategic (deliberate), Agressor (offensive)
        names = ["Vanilla", "Strategic", "Agressor"]
        model_names = cls.model_names()
        players = [
            # Pass None so Player.__init__ can pick up per-player temps from session_state
            Player(name, model_name, None)
            for name, model_name in zip(names, model_names)
        ]
        return cls(players)

    def turn_name(self) -> str:
        return f"Turn {self.turn}"

    def table(self) -> pd.DataFrame:
        """
        Build a dataframe of cumulative scores per turn for plotting.
        Missing future turns are padded with NaN so Streamlit charts render clean axes.
        """
        d = {}
        padding = [math.nan] * (11 - self.turn)
        for player in self.players:
            series = player.series[:] + padding
            d[player.name] = series[:11]
        return pd.DataFrame(data=d, index=range(11))

    @staticmethod
    def rankings() -> pd.DataFrame:
        """
        Create the leaderboard, delegating to the Game business object to handle this
        :return: a dataframe with the leaderboard info
        """
        df = Game.games_df()
        df = df.sort_values(by="Win %", ascending=False)
        supported_models = LLM.all_model_names()
        df = df[df["LLM"].isin(supported_models)]
        return df

    @staticmethod
    def latest() -> pd.DataFrame:
        """
        Create the table of last N games, delegating to the Game business object
        :return: a dataframe with the most recent results of games
        """
        return Game.latest_df()
