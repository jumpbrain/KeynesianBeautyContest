# Keynesian Beauty Contest

An interactive prototype for experimenting with repeated Keynesian beauty contests between LLM agents. Three LLM's play each match:

- **Vanilla** – a naive baseline.
- **Strategic** – a recursive (k-level) reasoner that iteratively anticipates opponents moves.
- **Agressor** – a very offensive player.

## Features

- Streamlit interface with one-click run controls.
- Sidebar controls for randomized or manual starters plus per-agent temperatures.
- Cumulative score chart.
- Model as a judge eval and summary of each round.

## Project Layout

- `app.py` – Streamlit entry point.
- `views/` – UI components (headers, sidebar, displays).
- `game/` – Arena orchestration, players, and referee logic.
- `prompting/` – System and user prompt templates for each persona.
- `models/` – Pydantic models and CSV/Mongo storage adapters.
- `tools/` – Maintenance utilities (e.g moves tidy script).
- `data/` – Created to hold move logs (`moves.csv`).

## Acknowledgements

Inspired by Edward Donner's Outsmart arena. This sandbox is built on the same idea of competitive LLM play. 

## License

This repository uses a custom license that allows personal/internal use but requires written permission before you redistribute the software. See `LICENSE` for full terms.

