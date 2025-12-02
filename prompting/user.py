from typing import List
from models.records import TurnRecord


TARGET_MULTIPLIER = 0.7


def _describe_players(other_names: List[str]) -> str:
    if len(other_names) == 1:
        return f"There is exactly 1 other player: {other_names[0]}."
    if len(other_names) > 1:
        others = ", ".join(other_names)
        return f"The other players are {others}."
    return "There are no other players."


def _instruction_block(name: str, messages_enabled: bool = True) -> str:
    lines = [
        "INSTRUCTIONS:",
        "- Output ONLY valid JSON (no commentary, no markdown).",
        "- Use exactly these keys: 'secret strategy', 'inner_thoughts', 'guess', 'public message'.",
        "- 'guess' must be a number between 0 and 100 inclusive. Decimals are allowed if justified.",
        "- 'inner_thoughts' must be an object containing a 'prediction' of what you expect the TARGET to be (or how opponents will guess) and a concise 'why'.",
        "- 'public message' is optional; use an empty string if you do not wish to broadcast anything.",
        "- Remember: the target each round is p * (average guess) with p = 0.7. Aim your guess to be closest to that target.",
    ]
    if name == "Vanilla":
        lines.append("- Vanilla focus: follow a naive, low-level strategy. Take others at face value and choose the simplest justified response.")
    elif name == "Agressor":
        lines.append("- Agressor focus: play the most offensive strategy. Push the average upward, favour high guesses, and keep pressure on every round.")
    else:
        lines.append("- Strategic focus: employ explicit k-level reasoning. Forecast iterative adjustments and document your level-k rationale.")
    return "\n".join(lines)


def first_turn(name: str, other_names: List[str], score: float, messages_enabled: bool = True) -> str:
    others_desc = _describe_players(other_names)
    instructions = _instruction_block(name, messages_enabled)
    example_guess = 35 if len(other_names) > 0 else 0
    if name == "Agressor":
        example_guess = 95
    example = (
        "Return JSON exactly in this shape (example):\n"
        "{\n"
        "  \"secret strategy\": \"Outline the private reasoning steps you'll follow\",\n"
        "  \"inner_thoughts\": {\n"
        "    \"prediction\": \"I expect the target to land near 35\",\n"
        "    \"why\": \"Players often converge near 0.7 times a mid-range guess; I expect mild level-k adjustments\"\n"
        "  },\n"
        f"  \"guess\": {example_guess},\n"
        "  \"public message\": \"Announcing a calm rationale builds credibility.\"\n"
        "}"
    )
    return (
        f"Your player name is {name}. {others_desc}\n\n"
        "Game: Keynes Beauty Contest. Each round every player guesses a number between 0 and 100.\n"
        "After all guesses, compute the average and multiply by 0.7. Whoever is closest to that target earns more points.\n\n"
        f"You currently have a total score of {score:.2f}.\n\n"
        f"{instructions}\n\n"
        f"{example}"
    )


def for_turn(
    name: str,
    other_names: List[str],
    other_scores: List[float],
    score: float,
    turn: int,
    records: List[TurnRecord],
    messages_enabled: bool = True,
) -> str:
    others_desc = _describe_players(other_names)
    instructions = _instruction_block(name, messages_enabled)

    prompt_parts = [
        f"Your player name is {name}. {others_desc}",
        "\nRECENT HISTORY:\n",
    ]
    if records:
        for record in records:
            prompt_parts.append(str(record) + "\n")
    else:
        prompt_parts.append("(No previous rounds yet.)\n")

    prompt_parts.append(
        f"Current turn: {turn}.\n"
        f"Your cumulative score: {score:.2f}.\n"
        "Opponents hold these scores:\n"
    )
    for other_name, other_score in zip(other_names, other_scores):
        prompt_parts.append(f"- {other_name}: {other_score:.2f}\n")

    prompt_parts.append("\nRemember: target = 0.7 * average(all guesses). Closest guess earns the biggest gain.\n\n")
    prompt_parts.append(instructions + "\n\n")
    prompt_parts.append(
        "Return JSON exactly in this shape (example):\n"
        "{\n"
        "  \"secret strategy\": \"Specify your private reasoning chain\",\n"
        "  \"inner_thoughts\": {\n"
        "    \"prediction\": \"I expect the target near 28\",\n"
        "    \"why\": \"Opponent signalled a lower level reasoning last round so I forecast a slight adjustment\"\n"
        "  },\n"
        "  \"guess\": 30,\n"
        "  \"public message\": \"Share an optional public justification or leave empty.\"\n"
        "}"
    )

    return "".join(prompt_parts)


def prompt(
    name: str,
    other_names: List[str],
    other_scores: List[float],
    score: float,
    turn: int,
    records: List[TurnRecord],
    messages_enabled: bool = True,
) -> str:
    if turn == 1:
        return first_turn(name, other_names, score, messages_enabled)
    return for_turn(name, other_names, other_scores, score, turn, records, messages_enabled)
