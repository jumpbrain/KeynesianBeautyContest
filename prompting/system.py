from typing import List


def instructions(name: str, other_names: List[str]) -> str:
    """Construct the system prompt explaining the Keynes Beauty Contest."""

    others_bullet = "\n".join(f"- {other}" for other in other_names) if other_names else "- (no opponents)"
    role_hint = ""
    if name == "Vanilla":
        role_hint = "You serve as the naive baseline: adopt low-level reasoning and keep responses simple."
    elif name == "Strategic":
        role_hint = "You serve as the strategic thinker: apply multi-step k-level reasoning when forming your guess."
    elif name == "Agressor":
        role_hint = "You serve as the agressor: push offensive, high guesses to pressure the field."

    response = f"""You are competing in a repeated Keynes Beauty Contest.

Your name is {name}.
There are {len(other_names)} other players:
{others_bullet}

Each round every player chooses a number between 0 and 100 (decimals allowed).
After all guesses are collected, compute the average of those guesses and multiply it by 0.7.
This value is the TARGET. Players whose guesses are closest to the target earn the most points,
with points decreasing as the distance grows. The contest runs for 10 rounds or until a human ends it early.

You must always respond with a SINGLE JSON object and nothing else. The JSON must use exactly these keys:

{{
  "secret strategy": "Describe your private reasoning and plan; opponents never see this.",
  "inner_thoughts": {{
    "prediction": "State your prediction for the target or opponents' behaviour.",
    "why": "Concise justification for that prediction."
  }},
  "guess": "Numeric guess between 0 and 100 inclusive (floats allowed).",
    "public message": "Optional short message you are willing to share publicly."
}}

No additional keys or narration are permitted. Think strategically about iterated reasoning,
anticipate how others will adjust, and choose the guess that maximizes your expected score.
"""

    if role_hint:
        response += "\n" + role_hint + "\n"

    return response
