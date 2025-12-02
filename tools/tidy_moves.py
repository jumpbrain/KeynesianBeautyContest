"""Normalize moves CSV files to the current Keynes Beauty Contest schema."""

from pathlib import Path
import argparse
import pandas as pd


def tidy_df(df: pd.DataFrame, MoveLogger) -> pd.DataFrame:
    df = df.copy()

    # Ensure all expected columns exist
    for column in MoveLogger.HEADERS:
        if column not in df.columns:
            df[column] = ""

    numeric_cols = [
        "guess",
        "applied_guess",
        "target",
        "distance",
        "score_delta",
        "prior_score",
        "post_score",
        "temperature",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    bool_cols = ["is_invalid", "repair_attempted"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])

    text_cols = ["system_prompt", "user_prompt", "repaired_response", "raw_response", "public_message"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    tidy = df[MoveLogger.HEADERS].copy()
    return tidy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default=None, help="Path to an existing moves.csv to tidy")
    parser.add_argument("--outfile", type=str, default=None, help="Output tidy csv path")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file with tidy output")
    args = parser.parse_args()

    try:
        from beauty.models.storage import MoveLogger, DEFAULT_MOVES_CSV  # type: ignore
    except Exception:
        from ..models.storage import MoveLogger, DEFAULT_MOVES_CSV  # type: ignore

    infile = Path(args.infile) if args.infile else Path(DEFAULT_MOVES_CSV)
    if not infile.exists():
        print(f"Input file not found: {infile}")
        return

    outfile = Path(args.outfile) if args.outfile else infile.parent / "moves_tidy.csv"
    if args.inplace:
        outfile = infile

    try:
        df = MoveLogger.load_df(infile)
    except Exception:
        df = pd.read_csv(infile, dtype=str, keep_default_na=False)

    tidy = tidy_df(df, MoveLogger)

    try:
        tidy.to_csv(outfile, index=False)
        print(f"Wrote tidy CSV with {len(tidy)} rows to: {outfile}")
    except Exception as exc:
        print(f"Failed to write tidy CSV: {exc}")


if __name__ == "__main__":
    main()
