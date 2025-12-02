# Keynesian Beauty Contest

An interactive Streamlit sandbox for experimenting with repeated Keynesian Beauty Contests between LLM agents. Three personas play every match:

- **Vanilla** – a naive baseline that anchors around straightforward best responses.
- **Strategic** – a k-level reasoner that iteratively anticipates opponents.
- **Agressor** – a red-team style agent that pushes offensive, high-impact guesses.

The UI exposes temperature controls, starter policies (random or manual), turn-by-turn internals, and live score tracking so you can observe how different prompting strategies evolve over ten rounds.

## Features

- Streamlit interface with one-click `Run Turn`, `Run Simulation`, and `Restart` controls.
- Player-specific colors on the cumulative score chart with integer turn ticks.
- Sidebar controls for randomized or manual starters plus per-agent temperatures.
- Automatic per-turn logging to `data/moves.csv` (created on demand) for post-game analysis.
- Prompt templates tailored to each persona so behaviour stays aligned during multi-round play.

## Prerequisites

- Python 3.11
- Streamlit-compatible API keys (e.g., OpenAI) configured as environment variables before launching the app.

You can store credentials in a local `.env` file (kept out of version control) with values such as:

```
OPENAI_API_KEY="..."
```

## Quick Start

```powershell
git clone <your repo url>
cd beauty
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local Streamlit URL provided in the terminal. The default session randomizes the starting player each round; toggle the sidebar to pick a manual starter if you need deterministic ordering.

## Project Layout

- `app.py` – Streamlit entry point.
- `views/` – UI components (headers, sidebar, displays).
- `game/` – Arena orchestration, players, and referee logic.
- `prompting/` – System and user prompt templates for each persona.
- `models/` – Pydantic models and CSV/Mongo storage adapters.
- `tools/` – Maintenance utilities such as the moves tidy script.
- `data/` – Created automatically to hold move logs (`moves.csv`).


## Acknowledgements

Inspired by Edward Donner's Outsmart arena. This sandbox is built on the same idea of competitive LLM play. 


## License

This repository uses a custom license that allows personal/internal use but requires written permission before you redistribute the software. See `LICENSE` for full terms.

