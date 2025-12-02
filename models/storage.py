import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


DEFAULT_MOVES_CSV = Path(__file__).resolve().parent.parent / "data" / "moves.csv"


def _ensure_dir(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


class MoveLogger:
    """Simple CSV appender/loader for per-turn move records."""

    HEADERS = [
        "run_date",
        "turn",
        "timestamp",
        "player",
        "model_name",
        "temperature",
        "strategy",
        "guess",
        "applied_guess",
        "target",
        "distance",
        "score_delta",
        "prior_score",
        "post_score",
        "public_message",
        "raw_response",
        "is_invalid",
        "system_prompt",
        "user_prompt",
        "repair_attempted",
        "repaired_response",
        "inner_prediction",
        "inner_why",
    ]

    @classmethod
    def append(cls, path: Path, record: Dict[str, Any]):
        p = Path(path or DEFAULT_MOVES_CSV)
        _ensure_dir(p)
        write_header = not p.exists()
        if not write_header:
            try:
                with p.open("r", encoding="utf-8", newline="") as existing_fh:
                    reader = csv.reader(existing_fh)
                    current_header = next(reader, [])
            except Exception:
                current_header = []

            if current_header != cls.HEADERS:
                existing_df = cls.load_df(p)
                with p.open("w", encoding="utf-8", newline="") as rewrite_fh:
                    writer = csv.writer(rewrite_fh)
                    writer.writerow(cls.HEADERS)
                    if not existing_df.empty:
                        for _, row in existing_df.iterrows():
                            writer.writerow([row.get(col, "") for col in cls.HEADERS])
                write_header = False

        with p.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(cls.HEADERS)

            def _to_str(value):
                if value is None:
                    return ""
                if isinstance(value, (dict, list)):
                    return json.dumps(value, ensure_ascii=False)
                return value

            out_row = [_to_str(record.get(h, "")) for h in cls.HEADERS]
            writer.writerow(out_row)

    @classmethod
    def log_turn(cls, csv_path: Path, run_date: str, turn: int, rec_obj) -> None:
        """Log a TurnRecord-like object to CSV. rec_obj may be a TurnRecord or similar."""
        path = Path(csv_path or DEFAULT_MOVES_CSV)
        data = {
            "run_date": run_date,
            "turn": turn,
            "timestamp": datetime.utcnow().isoformat(),
            "player": getattr(rec_obj, "name", ""),
            "model_name": getattr(rec_obj, "model_name", ""),
            "temperature": getattr(rec_obj, "temperature", ""),
            "strategy": getattr(rec_obj, "move", None) and getattr(rec_obj.move, "strategy", ""),
            "guess": getattr(rec_obj, "guess", ""),
            "applied_guess": getattr(rec_obj, "applied_guess", ""),
            "target": getattr(rec_obj, "target_value", ""),
            "distance": getattr(rec_obj, "distance_from_target", ""),
            "score_delta": getattr(rec_obj, "score_delta", ""),
            "prior_score": getattr(rec_obj, "prior_score", ""),
            "post_score": getattr(rec_obj, "post_score", ""),
            "public_message": getattr(rec_obj.move, "public_message", "") if getattr(rec_obj, "move", None) else "",
            "raw_response": getattr(rec_obj, "raw_response", ""),
            "is_invalid": getattr(rec_obj, "is_invalid_move", False),
            "system_prompt": getattr(rec_obj, "system_prompt", "")[:4000],
            "user_prompt": getattr(rec_obj, "user_prompt", "")[:4000],
            "repair_attempted": getattr(rec_obj, "repair_attempted", False),
            "repaired_response": getattr(rec_obj, "repaired_response", ""),
            "inner_prediction": getattr(rec_obj, "inner_prediction", "")
            or (
                getattr(rec_obj, "move", None)
                and getattr(rec_obj.move, "inner_thoughts", {}).get("prediction", "")
            ),
            "inner_why": getattr(rec_obj, "inner_why", "")
            or (
                getattr(rec_obj, "move", None)
                and getattr(rec_obj.move, "inner_thoughts", {}).get("why", "")
            ),
        }
        cls.append(path, data)

    @classmethod
    def load_df(cls, path: Path = None):
        import pandas as pd
        import csv

        p = Path(path or DEFAULT_MOVES_CSV)
        if not p.exists():
            return pd.DataFrame(columns=cls.HEADERS)

        rows = []
        with p.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return pd.DataFrame(columns=cls.HEADERS)

            for r in reader:
                # If row has more columns than header, merge extras into last column
                if len(r) > len(header):
                    r = r[: len(header) - 1] + [",".join(r[len(header) - 1 :])]
                # If row has fewer columns, pad with empty strings
                if len(r) < len(header):
                    r = r + [""] * (len(header) - len(r))
                rows.append(dict(zip(header, r)))

        # Build DataFrame
        df = pd.DataFrame(rows)

        # Normalize to current HEADERS: ensure all expected cols exist
        for h in cls.HEADERS:
            if h not in df.columns:
                df[h] = ""
        # Keep only HEADERS order
        df = df[cls.HEADERS]
        return df
