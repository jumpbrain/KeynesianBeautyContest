from typing import Optional, Dict, Any
from models.moves import Move


class TurnRecord:
    """Snapshot of a player's move and outcome for a single turn."""

    turn: int
    name: str
    is_invalid_move: bool
    move: Optional[Move] = None
    raw_response: str

    def __init__(self, name: str, turn: int, move=None, is_invalid_move=False, raw_response: str = ""):
        self.name = name
        self.turn = turn
        self.is_invalid_move = is_invalid_move
        self.system_prompt = ""
        self.user_prompt = ""
        self.model_name = ""
        self.temperature = None
        self.repair_attempted = False
        self.repaired_response = ""
        self.prior_score: Optional[float] = None
        self.post_score: Optional[float] = None
        self.inner_thoughts: Dict[str, Any] = {}
        self.inner_prediction: Optional[str] = None
        self.inner_why: Optional[str] = None
        self.guess: Optional[float] = None
        self.applied_guess: Optional[float] = None
        self.target_value: Optional[float] = None
        self.distance_from_target: Optional[float] = None
        self.score_delta: Optional[float] = None
        self.move = move
        self.raw_response = raw_response

    def __repr__(self) -> str:
        """
        Convert this TurnRecord into text; this is used to describe historic moves when making the prompt
        :return: a string to represent this instance
        """
        result = f"Recap of Turn {self.turn}\n\n"
        result += "Your actions:\n"
        if self.is_invalid_move:
            result += "You provided invalid JSON, so your move was not processed\n"
            if self.raw_response:
                result += "Raw response:\n"
                excerpt = self.raw_response
                if len(excerpt) > 800:
                    excerpt = excerpt[:800] + "..."
                result += excerpt + "\n"
        else:
            guess_display = self.applied_guess if self.applied_guess is not None else getattr(self.move, "guess", None)
            result += f"Your secret strategy: {self.move.strategy}\n"
            try:
                it = (self.inner_thoughts or getattr(self.move, "inner_thoughts", {}) or {})
                if it and isinstance(it, dict):
                    prediction = it.get("prediction") or self.inner_prediction
                    why = it.get("why") or self.inner_why
                    if prediction:
                        result += f"Prediction: {prediction}\n"
                        if why:
                            result += f"Reason: {why}\n"
            except Exception:
                pass
            if guess_display is not None:
                result += f"You guessed: {guess_display:.2f}\n"
            public_message = getattr(self.move, "public_message", "")
            if public_message:
                result += f"Public message: {public_message}\n"

        result += "\nResults of the turn:\n"
        if self.target_value is not None:
            result += f"Target (p * average guess): {self.target_value:.2f}\n"
        if self.distance_from_target is not None:
            result += f"Distance from target: {self.distance_from_target:.2f}\n"
        if self.score_delta is not None:
            result += f"Score change: {self.score_delta:.2f}\n"
        if self.post_score is not None:
            result += f"Total score: {self.post_score:.2f}\n"

        result += "\n"
        return result
