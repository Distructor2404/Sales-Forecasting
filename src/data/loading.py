"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and preprocess the raw sales dataset."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    return df


def get_active_pairs(df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Return store-SKU pairs with observations on or before as_of_date."""
    recent = df[df["date"] <= as_of_date]
    pairs = recent.groupby(["store_id", "sku_id"], as_index=False).agg(
        last_date=("date", "max"),
        n_days=("date", "count"),
    )
    return pairs
