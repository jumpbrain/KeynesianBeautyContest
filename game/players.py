from typing import List, Dict, Any, Self
from interfaces.llms import LLM
from prompting.system import instructions
from prompting.user import prompt
from models.records import TurnRecord


class Player:
    """
    Individual competitor in the Beauty arena using an underlying LLM interface
    """

    name: str
    llm: LLM
    others: List[Self]
    history: Dict[str, Any]
    score: float
    records: List[TurnRecord]
    is_dead: bool
    is_winner: bool
    series: List[float]

    MAX_TOKENS = 600

    def __init__(self, name: str, model_name: str, temperature: float = None):
        """
        Create a new instance of Player
        :param name: The Player's name, as the others will address them
        :param model_name: Which LLM model to use
        :param temperature: The temperature setting for the model, so that different temp models can compete
                            If None, attempt to read per-player temperature from Streamlit `st.session_state['player_temps']`.
        """
        self.name = name
        # Determine the temperature to use: explicit argument > session setting > default 0.7
        temp_used = temperature
        if temp_used is None:
            try:
                import streamlit as st

                temp_used = st.session_state.get("player_temps", {}).get(name, None)
            except Exception:
                temp_used = None
        if temp_used is None:
            temp_used = 0.7

        try:
            self.llm = LLM.for_model_name(model_name, temp_used)
        except Exception as e:
            # Fallback to a known OpenAI model if requested model cannot be instantiated
            # This prevents crashes when a synthetic model name (like gpt-5-strategic-k2)
            # cannot be used directly by the underlying client.
            from logging import getLogger

            getLogger(__name__).warning(
                f"Could not instantiate model '{model_name}': {e}. Falling back to 'gpt-5-mini'."
            )
            self.llm = LLM.for_model_name("gpt-5-mini", temp_used)
        self.history = {}
        self.score = 0.0
        self.prior_score = 0.0
        self.series = [0.0]
        self.others = []  # this will be initialized during Arena construction
        self.records = []
        self.is_dead = False
        self.is_winner = False

    def __repr__(self) -> str:
        """
        :return: a String to represent this player
        """
        return f"[Player {self.name} with score {self.score:.2f} using {self.llm}]"

    def system_prompt(self) -> str:
        """
        :return: a System Prompt to be sent to the LLM
        """
        other_names = [other.name for other in self.others]
        return instructions(self.name, other_names)

    def user_prompt(self, turn: int) -> str:
        """
        :return: a User prompt to instruct the LLM for this player for this turn
        """
        other_names = [other.name for other in self.others]
        other_scores = [other.score for other in self.others]
        # Determine whether private messaging is enabled in the UI (default True)
        try:
            import streamlit as st

            messages_enabled = st.session_state.get("messaging_enabled", True)
        except Exception:
            messages_enabled = True

        return prompt(
            self.name, other_names, other_scores, self.score, turn, self.records, messages_enabled
        )

    def make_move(self, turn: int):
        """
        Carry out a turn by interfacing with my LLM
        :param turn: which turn number we are on
        :return: the response from the LLM
        """
        system_prompt = self.system_prompt()
        user_prompt = self.user_prompt(turn)
        response = self.llm.send(system_prompt, user_prompt, self.MAX_TOKENS)
        # Return the response plus metadata so caller can log prompts and parameters
        return {
            "response": response,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model_name": getattr(self.llm, "model_name", ""),
            "temperature": getattr(self.llm, "temperature", None),
        }

    def report(self) -> str:
        """
        Create a report of this player
        :return:
        """
        result = f"Player name: {self.name}<br/>"
        result += f"Model: {self.llm.model_name}<br/>"
        result += f"Temperature: {self.llm.temperature}<br/><br/>"
        for turn_record in self.records:
            result += str(turn_record).replace("\n", "<br/>")
            result += "<br/>"
        return result

    def kill(self) -> None:
        """
        This player has died - update the status
        """
        self.is_dead = True
