from typing import List, Dict, Callable, Tuple
import json
import re
import logging
from game.players import Player
from models.moves import Move
from models.records import TurnRecord
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

TARGET_MULTIPLIER = 0.7


class Referee:

    players: List[Player]
    turn: int
    records: Dict[str, TurnRecord]
    player_names = List[str]
    player_map: Dict[str, Player]
    alliances = List[str]

    def __init__(self, players: List[Player], turn: int, run_date: str = None):
        """
        Initialize this instance
        :param players: list of players
        :param turn: turn number
        """
        self.players = players
        self.turn = turn
        self.records = {}
        self.player_names = [player.name for player in players]
        self.player_map = {player.name: player for player in players}
        self.run_date = run_date

    def do_turn_for_player(self, player: Player) -> TurnRecord:
        """
        Carry out a turn for this player whilst handling any exceptions raised
        :param player: the player being processed
        :return: a TurnRecord that wraps the output from the model, including whether it was valid
        """
        response = ""
        try:
            response = player.make_move(self.turn)
            # response is now a dict with response and metadata
            resp_text = response.get("response", "")
            system_prompt = response.get("system_prompt", "")
            user_prompt = response.get("user_prompt", "")
            model_name = response.get("model_name", "")
            temperature = response.get("temperature", None)
            try:
                move = self.parse_response(resp_text)
                logger.info(f"Turn {self.turn} received OK from {player}")
                rec = TurnRecord(player.name, self.turn, move=move, raw_response=resp_text)
                rec.system_prompt = system_prompt
                rec.user_prompt = user_prompt
                rec.model_name = model_name
                rec.temperature = temperature
                rec.prior_score = getattr(player, "prior_score", None)
                # capture inner thoughts if provided
                try:
                    it = getattr(move, "inner_thoughts", {}) or {}
                    rec.inner_thoughts = it
                    if isinstance(it, dict):
                        rec.inner_prediction = it.get("prediction")
                        rec.inner_why = it.get("why")
                except Exception:
                    pass
                # store the raw guess (clamped by Move validation)
                try:
                    rec.guess = float(move.guess)
                    rec.applied_guess = float(move.guess)
                except Exception:
                    rec.guess = None
                    rec.applied_guess = None

                return rec
            except Exception as parse_e:
                logger.warning(f"Initial response from {player} could not be parsed: {parse_e}")
                logger.debug(f"Raw response: {response}")
                # Attempt to repair the response by asking the same model to reformat
                try:
                    repaired = self.repair_response(player, resp_text)
                    move = self.parse_response(repaired)
                    logger.info(f"Turn {self.turn} repaired and accepted from {player}")
                    rec = TurnRecord(player.name, self.turn, move=move, raw_response=repaired)
                    rec.system_prompt = system_prompt
                    rec.user_prompt = user_prompt
                    rec.model_name = model_name
                    rec.temperature = temperature
                    rec.repair_attempted = True
                    rec.repaired_response = repaired
                    rec.prior_score = getattr(player, "prior_score", None)
                    try:
                        it = getattr(move, "inner_thoughts", {}) or {}
                        rec.inner_thoughts = it
                        if isinstance(it, dict):
                            rec.inner_prediction = it.get("prediction")
                            rec.inner_why = it.get("why")
                    except Exception:
                        pass
                    try:
                        rec.guess = float(move.guess)
                        rec.applied_guess = float(move.guess)
                    except Exception:
                        rec.guess = None
                        rec.applied_guess = None

                    return rec
                except Exception as repair_e:
                    logger.error(f"Repair attempt failed for {player}: {repair_e}")
                    logger.error(f"Response received was:\n{resp_text}")
                    rec = TurnRecord(player.name, self.turn, is_invalid_move=True, raw_response=resp_text)
                    rec.system_prompt = system_prompt
                    rec.user_prompt = user_prompt
                    rec.model_name = model_name
                    rec.temperature = temperature
                    return rec
        except Exception as e:
            logger.error(f"Exception while processing response from {player}")
            logger.error(e)
            logger.error(f"Response received was:\n{response}")
            rec = TurnRecord(player.name, self.turn, is_invalid_move=True, raw_response=response)
            return rec

    def repair_response(self, player: Player, original_response: str) -> str:
        """
        Try to coax the LLM into returning a valid JSON move by sending a short
        reformatting prompt that includes the original response for context.
        Returns the repaired response text (expected to contain a JSON object).
        """
        try:
            repair_user = (
                """
Your previous response could not be parsed as valid JSON for this Keynes Beauty Contest turn.
Here is what you returned:

"""
                + original_response
                + """

Return ONLY one JSON object with exactly these keys: 'secret strategy', 'inner_thoughts', 'guess', 'public message'.
- 'guess' must be a number between 0 and 100.
- 'inner_thoughts' must contain 'prediction' and 'why'.
Do NOT add explanatory text before or after the JSON object.
"""
            )
            repaired = player.llm.send(player.system_prompt(), repair_user, player.MAX_TOKENS // 4)
            return repaired
        except Exception as e:
            logger.error(f"Error while attempting repair with model for player {player.name}: {e}")
            # As a fallback, attempt to have another player (helper) reformat the JSON
            for helper in self.players:
                if helper.name == player.name:
                    continue
                try:
                    helper_prompt = (
                        """
Another player produced an output that could not be parsed as JSON.
Please reformat the following text into a single valid JSON object with the keys:
'secret strategy', 'inner_thoughts' (with 'prediction' and 'why'), 'guess', 'public message'.
"""
                        + original_response
                        + "\n\nReturn ONLY the JSON object."
                    )
                    repaired = helper.llm.send(helper.system_prompt(), helper_prompt, helper.MAX_TOKENS // 4)
                    logger.info(f"Helper {helper.name} succeeded in reformatting response for {player.name}")
                    return repaired
                except Exception:
                    logger.debug(f"Helper {helper.name} failed to repair response for {player.name}")
            # nothing worked
            raise

    def player_with_name(self, name: str) -> Player:
        """
        Return the player with the given name
        :param name: the name of a player
        :return: the player object
        """
        player = self.player_map[name]
        if player:
            return player
        else:
            raise ValueError(f"Failed to find player with name {name}")

    def do_turn(self, progress: ProgressCallback) -> None:
        """
        This is called by an Arena object to run a Turn
        First get each Player to make a move using ThreadPoolExecutor to run in parallel
        Then evaluate each Player in turn
        :param progress: a callback on which to report progress that will be reflected in the UI
        :return:
        """
        progress(0, "Players are thinking..")
        responded = []
        with ThreadPoolExecutor(max_workers=len(self.players)) as e:
            for record in e.map(self.do_turn_for_player, self.players):
                player = self.player_with_name(record.name)
                responded.append(record.name)
                prog = len(responded) / len(self.players)
                progress(prog, f"{', '.join(responded)} responded..")
                self.records[record.name] = record
                player.records.append(record)
        progress(1.0, "Finishing up..")
        self.handle_turn()

    def handle_turn(self) -> None:
        """Evaluate guesses, update scores, and log turn outcomes."""
        # Collect valid guesses
        valid_records = [
            rec
            for rec in self.records.values()
            if not rec.is_invalid_move and rec.applied_guess is not None
        ]

        target = None
        if valid_records:
            guesses = [float(rec.applied_guess) for rec in valid_records]
            avg_guess = sum(guesses) / len(guesses)
            target = TARGET_MULTIPLIER * avg_guess

        for name, record in self.records.items():
            player = self.player_map[name]
            if record.is_invalid_move or record.applied_guess is None or target is None:
                record.target_value = target
                record.distance_from_target = None if target is None else abs((record.applied_guess or 0.0) - target)
                record.score_delta = 0.0
                record.post_score = player.score
                continue

            guess = float(record.applied_guess)
            # Ensure guess is in bounds even if model slipped something odd
            guess = max(0.0, min(100.0, guess))
            record.applied_guess = guess
            record.target_value = target
            distance = abs(guess - target)
            record.distance_from_target = distance
            score_delta = max(0.0, 100.0 - distance)
            record.score_delta = score_delta
            player.score += score_delta
            record.post_score = player.score

        # After scores are updated, persist turn records
        try:
            from models.storage import MoveLogger

            for record in self.records.values():
                MoveLogger.log_turn(None, self.run_date or "", self.turn, record)
        except Exception:
            logger.debug("Failed to log move to CSV")


    def parse_response(self, response: str) -> Tuple[Move, bool]:
        """
        Convert a text response into a Move object
        :param response: the text returned from an LLM
        :return: a Move object
        """
        import ast
        # Extract the first {...} block
        first = response.find("{")
        last = response.rfind("}")
        if first == -1 or last == -1 or last <= first:
            raise ValueError("No JSON object found in response")
        snippet = response[first : last + 1]

        # Try a sequence of parsers/sanitizers to be robust to common model formatting issues
        # 1) direct json.loads
        try:
            response_dict = json.loads(snippet)
        except Exception:
            # 2) try ast.literal_eval to handle Python-style dicts with single quotes
            try:
                python_obj = ast.literal_eval(snippet)
                # ensure it's a dict
                if isinstance(python_obj, dict):
                    response_dict = python_obj
                else:
                    raise ValueError("literal_eval did not return a dict")
            except Exception:
                # 3) attempt basic cleanups: replace smart quotes and remove trailing commas
                cleaned = snippet
                # smart quotes to normal
                cleaned = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
                # remove trailing commas before } or ]
                cleaned = re.sub(r",\s*(\}|\])", r"\1", cleaned)
                # try json.loads again
                try:
                    response_dict = json.loads(cleaned)
                except Exception as e:
                    logger.debug(f"Sanitization failed, snippet was: {snippet}")
                    raise ValueError(f"Failed to parse JSON from response: {e}")

        move = Move(**response_dict)
        return move
