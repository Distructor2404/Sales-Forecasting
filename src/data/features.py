"""Leakage-aware feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


CATEGORICAL_COLS = [
    "store_id",
    "country",
    "city",
    "channel",
    "sku_id",
    "category",
    "subcategory",
    "brand",
    "supplier_id",
]

NUMERIC_FEATURE_COLS = [
    "year",
    "month",
    "day",
    "weekofyear",
    "weekday",
    "is_weekend",
    "is_holiday",
    "temperature",
    "rain_mm",
    "latitude",
    "longitude",
    "list_price",
    "discount_pct",
    "promo_flag",
    "stock_on_hand",
    "stock_out_flag",
    "lead_time_days",
    "purchase_cost",
    "margin_pct",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "effective_price",
    "promo_x_weekend",
    "rain_heavy",
    "temp_bucket",
]


class FeatureEncoder:
    """Fit label encoders on training data only."""

    def __init__(self, categorical_cols: list[str] | None = None):
        self.categorical_cols = categorical_cols or CATEGORICAL_COLS
        self.encoders: dict[str, LabelEncoder] = {}

    def fit(self, df: pd.DataFrame) -> FeatureEncoder:
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, le in self.encoders.items():
            values = out[col].astype(str)
            known = set(le.classes_)
            mapped = values.where(values.isin(known), "__unknown__")
            if "__unknown__" not in le.classes_:
                le.classes_ = np.append(le.classes_, "__unknown__")
            out[f"{col}_enc"] = le.transform(mapped)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def encoded_cols(self) -> list[str]:
        return [f"{c}_enc" for c in self.categorical_cols]


def _demand_for_features(series: pd.Series, stock_out: pd.Series) -> pd.Series:
    """Observed demand with stockout censoring handled via expanding mean imputation."""
    observed = series.copy().astype(float)
    mask = stock_out.astype(bool)
    if mask.any():
        non_censored = observed.where(~mask)
        fill_value = non_censored.expanding(min_periods=1).mean().shift(1)
        observed = observed.where(~mask, fill_value)
    return observed.fillna(0.0)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_price_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["effective_price"] = out["list_price"] * (1 - out["discount_pct"])
    out["promo_x_weekend"] = out["promo_flag"] * out["is_weekend"]
    return out


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rain_heavy"] = (out["rain_mm"] >= out["rain_mm"].median()).astype(int)
    out["temp_bucket"] = pd.cut(
        out["temperature"],
        bins=[-np.inf, 5, 15, 25, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    lag_days: list[int],
    rolling_windows: list[int],
    target: str = "units_sold",
) -> pd.DataFrame:
    """Compute per-series lag and rolling stats using only past observations."""
    out = df.copy()
    out["_demand_feat"] = out["units_sold"].astype(float)
    grouped = out.groupby(group_cols, sort=False)["_demand_feat"]

    for lag in lag_days:
        out[f"lag_{lag}"] = grouped.shift(lag)

    for window in rolling_windows:
        out[f"roll_mean_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        out[f"roll_std_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).std()
        ).fillna(0.0)

    out["days_since_stockout"] = out.groupby(group_cols, sort=False)["stock_out_flag"].transform(
        lambda s: s.eq(1).groupby((s != s.shift()).cumsum()).cumcount().where(s.eq(0), 0)
    )

    promo_group = out.groupby(group_cols, sort=False)["promo_flag"]
    out["promo_streak"] = promo_group.transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    ) * out["promo_flag"]

    out.drop(columns=["_demand_feat"], inplace=True)
    return out


def build_features(
    df: pd.DataFrame,
    group_cols: list[str],
    lag_days: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    encoder: FeatureEncoder | None = None,
    fit_encoder: bool = False,
) -> tuple[pd.DataFrame, FeatureEncoder | None]:
    lag_days = lag_days or [1, 7, 14, 28]
    rolling_windows = rolling_windows or [7, 14, 28]

    out = add_calendar_features(df)
    out = add_price_promo_features(out)
    out = add_weather_features(out)
    out = add_lag_rolling_features(out, group_cols, lag_days, rolling_windows)

    if fit_encoder:
        encoder = FeatureEncoder().fit(out)
    if encoder is not None:
        out = encoder.transform(out)

    return out, encoder


def get_ml_feature_columns(encoder: FeatureEncoder | None) -> list[str]:
    lag_roll = []
    for lag in [1, 7, 14, 28]:
        lag_roll.append(f"lag_{lag}")
    for window in [7, 14, 28]:
        lag_roll.extend([f"roll_mean_{window}", f"roll_std_{window}"])
    extra = ["days_since_stockout", "promo_streak"]
    encoded = encoder.encoded_cols if encoder else []
    return NUMERIC_FEATURE_COLS + lag_roll + extra + encoded
