"""
This module contains the interfaces to external LLMs;
There's an abstract base class LLM that can be subclassed to provide an interface to a model.
The class method LLM.for_model_name creates an instance of a subclass to interact with the API
This module should have no knowledge of the game itself.
"""

import os
import json
import re
from abc import ABC
from typing import Any, Dict, Self, List, Type
from openai import OpenAI
import anthropic
from groq import Groq


ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROK_BASE_URL = "https://api.x.ai/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


class LLM(ABC):
    """
    An abstract base class for LLMs
    Use LLM.for_model_name() to instantiate the appropriate subclass, then communicate with send()
    """

    model_names = []
    model_name: str
    temperature: float
    client: Any

    def __init__(self, model_name, temperature=1.0):
        self.model_name = model_name
        self.temperature = temperature
        self.setup_client()

    def setup_client(self):
        """
        Implemented by subclasses
        """
        pass

    def send(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """
        Implemented by subclasses
        :param system_prompt: The system prompt passed to the LLM
        :param user_prompt: The user prompt passed to the LLM
        :param max_tokens: Maximum number of tokens
        :return: the response from the LLM
        """
        pass

    def __repr__(self) -> str:
        """
        :return: A string version of the receiver
        """
        return f"<LLM {self.model_name} with temnp={self.temperature}>"

    @classmethod
    def model_map(cls) -> Dict[str, Type[Self]]:
        """
        Generate a mapping of Model Names to LLM classes, by looking at all subclasses of this one
        :return: a mapping dictionary from model name to LLM subclass
        """
        mapping = {}
        for llm in cls.__subclasses__():
            for model_name in llm.model_names:
                mapping[model_name] = llm
        return mapping

    @classmethod
    def for_model_name(cls, model_name: str, temperature=0.7) -> Self:
        """
        Given a particular model name, instantiate one of the subclasses of the receiver and initialize it
        :param model_name: The name of the model to be communicated with
        :param temperature: The temperature to be used in this model
        :return: an initialized instance of an LLM subclass
        """
        mapping = cls.model_map()
        # Exact match first
        if model_name in mapping:
            return mapping[model_name](model_name, temperature)

        # Otherwise, find the longest-registered name that is a prefix of the requested model_name
        best_match = None
        best_len = -1
        for registered_name, registered_class in mapping.items():
            if model_name.startswith(registered_name):
                if len(registered_name) > best_len:
                    best_len = len(registered_name)
                    best_match = registered_class

        if best_match is not None:
            return best_match(model_name, temperature)

        # As a final attempt, allow registered names that start with the requested value
        for registered_name, registered_class in mapping.items():
            if registered_name.startswith(model_name):
                if len(registered_name) > best_len:
                    best_len = len(registered_name)
                    best_match = registered_class

        if best_match is not None:
            return best_match(model_name, temperature)

        raise ValueError(f"Unknown model name: {model_name}")

    @classmethod
    def all_model_names(cls) -> List[str]:
        """
        :return: a list of names of all the models supported
        """
        return list(cls.model_map().keys())


class GPT(LLM):
    model_names = [
        "gpt-5",
        "gpt-5-nano",
        "gpt-5-mini",
    ]

    def setup_client(self):
        self.client = OpenAI()

    def send(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """
        Implementation for OpenAI / GPT
        :param system_prompt: The system prompt passed to the LLM
        :param user_prompt: The user prompt passed to the LLM
        :param max_tokens: Maximum number of tokens
        :return: the response from the LLM
        """
        effort = "low" if "gpt-5" in self.model_name else None
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            reasoning_effort=effort,
        )
        return completion.choices[0].message.content


class StrategicGPT(LLM):
    """
    An OpenAI-powered strategic evaluator that generates candidate moves then
    asks the model to evaluate them using a (configurable) k-level thinking
    instruction. The final returned value is the chosen candidate JSON string.
    This class keeps the game-specific logic out of the LLM layer by taking
    the same `system_prompt` and `user_prompt` as the vanilla GPT and
    wrapping them with instructions to propose and evaluate candidates.
    """

    model_names = [
        "gpt-5-strategic",  # default k
    ]

    def setup_client(self):
        self.client = OpenAI()

    def _parse_k(self) -> int:
        """Parse k-level from model_name if provided (e.g. gpt-5-strategic-k2)."""
        m = re.search(r"-k(\d+)", self.model_name)
        if m:
            return int(m.group(1))
        return 2

    def _extract_json(self, text: str) -> str:
        """Attempt to extract the first JSON object from text."""
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return text[first : last + 1]
        # fallback: try to load the whole text
        try:
            json.loads(text)
            return text
        except Exception:
            raise ValueError("Could not extract JSON from model response")

    def send(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        k = self._parse_k()

        # Determine a real API model name to call. The Strategic model_name may be a
        # synthetic alias like 'gpt-5-strategic-k2' which isn't an actual OpenAI model
        # identifier. Map that to a concrete model for API calls.
        # If this is a strategic-mode alias, always call the stable `gpt-5-mini`
        # API model for now. We'll implement a dedicated strategic engine later.
        if "strategic" in self.model_name:
            api_model = "gpt-5-mini"
        elif "gpt-5-mini" in self.model_name:
            api_model = "gpt-5-mini"
        elif "gpt-5-nano" in self.model_name:
            api_model = "gpt-5-nano"
        elif "gpt-5" in self.model_name:
            api_model = "gpt-5"
        else:
            api_model = "gpt-5-mini"

        # Step 1: ask the model to propose N candidate moves in the required JSON format
        propose_prompt = (
            user_prompt
            + "\n\nPlease propose 3 alternative candidate moves for the current turn."
            + " For each candidate, return a JSON object in the exact move format required by the game."
            + " Precede each JSON object with a short (1-2 sentence) private rationale."
            + " Return the three JSON objects separated clearly."
        )

        propose_resp = self.client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": propose_prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        propose_text = propose_resp.choices[0].message.content

        # Extract all JSON objects from the proposal text
        json_objects = re.findall(r"\{[\s\S]*?\}", propose_text)
        if not json_objects:
            # fallback: treat full response as single JSON
            json_objects = [self._extract_json(propose_text)]

        # Step 2: ask the model to evaluate each candidate with k-level thinking
        eval_prompt = (
            "You are an impartial evaluator. Given the current game context below,"
            " evaluate the following candidate moves and assign each a numeric score"
            " representing the expected final score advantage for the proposing player after"
            f" simulating up to {k} levels of reasoning (i.e. how opponents might respond).\n\n"
        )
        eval_prompt += "GAME CONTEXT:\n" + system_prompt + "\n" + user_prompt + "\n\n"
        eval_prompt += "CANDIDATES:\n"
        for i, j in enumerate(json_objects):
            eval_prompt += f"Candidate {i+1}:\n{j}\n\n"
        eval_prompt += (
            "For each candidate, provide a short rationale (1-2 sentences) and then a single numeric score on a"
            " line prefixed with `SCORE:`. Finally, state which candidate is best (by number) and output the"
            " full JSON of the winning candidate only."
        )

        eval_resp = self.client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": eval_prompt},
            ],
            temperature=max(0.0, self.temperature - 0.2),
            max_tokens=max_tokens,
        )

        eval_text = eval_resp.choices[0].message.content

        # Attempt to extract the winning JSON
        try:
            winning_json = self._extract_json(eval_text)
            return winning_json
        except Exception:
            # Fallback: return the highest-scored candidate by searching for 'SCORE:' labels
            scores = re.findall(r"SCORE:\s*([-+]?[0-9]*\.?[0-9]+)", eval_text)
            if scores and len(scores) == len(json_objects):
                # pick highest
                nums = [float(s) for s in scores]
                idx = int(max(range(len(nums)), key=lambda i: nums[i]))
                return json_objects[idx]
            # As last resort, return first candidate
            return json_objects[0]


class Claude(LLM):
    model_names = [
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ]

    def setup_client(self):
        self.client = anthropic.Anthropic()

    def send(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """
        Implementation for Anthropic / Claude
        :param system_prompt: The system prompt passed to the LLM
        :param user_prompt: The user prompt passed to the LLM
        :param max_tokens: Maximum number of tokens
        :return: the response from the LLM
        """
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=0.5,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        return message.content[0].text


# class Gemini(LLM):
#     model_names = ["gemini-1.0-pro", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"]

#     def setup_client(self):
#         google.generativeai.configure()
#         self.client = google.generativeai.GenerativeModel(self.model_name)

#     def send(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
#         """
#         Implementation for Google / Gemini
#         :param system_prompt: The system prompt passed to the LLM
#         :param user_prompt: The user prompt passed to the LLM
#         :param max_tokens: Maximum number of tokens
#         :return: the response from the LLM
#         """
#         words = int(max_tokens * 0.75)
#         message = "First, here is a System Message to set context and instructions:\n\n"
#         message += system_prompt + "\n\n"
#         message += f"Now here is the User's Request - please respond in under {words} words:\n\n"
#         message += user_prompt + "\n"
#         response = self.client.generate_content(message)
#         first_candidate = response.candidates[0]

#         if first_candidate.content.parts:
#             myanswer1 = response.candidates[0].content.parts[0].text
#             return myanswer1
#         raise ValueError("Could not parse response from Gemini")


class Grok(LLM):
    model_names = ["grok-4", "grok-4-fast"]

    def setup_client(self):
        self.client = OpenAI(api_key=os.getenv("GROK_API_KEY"), base_url=GROK_BASE_URL)

    def send(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """
        Implementation for OpenAI / GPT
        :param system_prompt: The system prompt passed to the LLM
        :param user_prompt: The user prompt passed to the LLM
        :param max_tokens: Maximum number of tokens
        :return: the response from the LLM
        """
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content


class Gemini(LLM):
    model_names = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    def setup_client(self):
        self.client = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url=GEMINI_BASE_URL)

    def send(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """
        Implementation for OpenAI / GPT
        :param system_prompt: The system prompt passed to the LLM
        :param user_prompt: The user prompt passed to the LLM
        :param max_tokens: Maximum number of tokens
        :return: the response from the LLM
        """
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content


class GroqAPI(LLM):
    """
    A class to act as an interface to the remote AI, in this case Groq
    """

    model_names = [
        "openai/gpt-oss-120b",
    ]

    def setup_client(self):
        self.client = Groq()

    def send(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """
        Implementation for Groq
        :param system_prompt: The system prompt passed to the LLM
        :param user_prompt: The user prompt passed to the LLM
        :param max_tokens: Maximum number of tokens
        :return: the response from the LLM
        """
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content
