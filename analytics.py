import pandas as pd


def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def compute_brier(df: pd.DataFrame) -> pd.DataFrame:
    """For the beauty contest, report distance-based accuracy metrics per player."""
    df = _ensure_df(df)
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["distance"] = pd.to_numeric(df.get("distance"), errors="coerce")
    df = df.dropna(subset=["distance"])
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby("player").agg(
        mean_abs_distance=("distance", "mean"),
        mean_squared_distance=("distance", lambda x: (x ** 2).mean()),
    ).reset_index()
    summary.rename(columns={
        "mean_abs_distance": "mean_abs_distance",
        "mean_squared_distance": "mean_squared_distance",
    }, inplace=True)
    return summary


def compute_regret(df: pd.DataFrame) -> pd.DataFrame:
    """Report average score delta per player (positive means consistent accuracy)."""
    df = _ensure_df(df)
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["score_delta"] = pd.to_numeric(df.get("score_delta"), errors="coerce").fillna(0.0)
    summary = df.groupby("player").agg(mean_score_delta=("score_delta", "mean")).reset_index()
    return summary


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate beauty-contest stats: mean guess, std dev, final score, invalid rate."""
    df = _ensure_df(df)
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["guess"] = pd.to_numeric(df.get("guess"), errors="coerce")
    df["post_score"] = pd.to_numeric(df.get("post_score"), errors="coerce")
    df["is_invalid"] = df.get("is_invalid", False).astype(bool)

    grouped = df.groupby("player").agg(
        turns=("turn", "nunique"),
        mean_guess=("guess", "mean"),
        std_guess=("guess", "std"),
        final_score=("post_score", "max"),
        invalid_rate=("is_invalid", "mean"),
    ).reset_index()
    return grouped
