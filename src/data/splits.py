"""Temporal train/validation/test splits."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TemporalSplit:
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def mask(self, dates: pd.Series, split: str) -> pd.Series:
        if split == "train":
            return dates <= self.train_end
        if split == "val":
            return (dates >= self.val_start) & (dates <= self.val_end)
        if split == "test":
            return (dates >= self.test_start) & (dates <= self.test_end)
        raise ValueError(f"Unknown split: {split}")


def create_temporal_split(
    df: pd.DataFrame,
    date_col: str = "date",
    test_days: int = 14,
    val_days: int = 28,
) -> TemporalSplit:
    """Create leakage-safe temporal boundaries."""
    max_date = df[date_col].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=val_days - 1)
    train_end = val_start - pd.Timedelta(days=1)

    return TemporalSplit(
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=max_date,
    )
