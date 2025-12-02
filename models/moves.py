from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any


class Move(BaseModel):
    """Structured move for the Keynes Beauty Contest turn."""

    strategy: str = Field(alias="secret strategy")
    guess: float = Field(alias="guess")
    # Optional inner thoughts reported by the agent: expects an object with keys 'prediction' and 'why'.
    inner_thoughts: Dict[str, Any] = Field(default_factory=dict, alias="inner_thoughts")
    # Allow an optional public justification that can be shown to humans.
    public_message: str = Field(default="", alias="public message")

    @field_validator("guess")
    @classmethod
    def clamp_guess(cls, value: float) -> float:
        """Clamp guesses into the allowed 0-100 interval."""
        try:
            numeric = float(value)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ValueError("guess must be numeric") from exc
        if numeric < 0:
            return 0.0
        if numeric > 100:
            return 100.0
        return numeric
