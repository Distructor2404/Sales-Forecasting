"""Tree-boosting baselines: LightGBM, XGBoost, CatBoost."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.data.splits import TemporalSplit


@dataclass
class TreeBoostConfig:
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50


LightGBMConfig = TreeBoostConfig
XGBoostConfig = TreeBoostConfig
CatBoostConfig = TreeBoostConfig

FUTURE_KNOWN_COLS = [
    "promo_flag",
    "discount_pct",
    "list_price",
    "effective_price",
    "is_weekend",
    "is_holiday",
    "weekday",
    "month",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "temperature",
    "rain_mm",
    "rain_heavy",
    "temp_bucket",
]


def _build_direct_frame(
    df: pd.DataFrame,
    origin_mask: pd.Series,
    feature_cols: list[str],
    target: str,
    horizon: int,
    max_target_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Vectorized direct multi-horizon dataset (origin features + future known inputs)."""
    group_cols = ["store_id", "sku_id"]
    origins = df.loc[origin_mask].copy()
    grouped = df.groupby(group_cols, sort=False)

    chunks = []
    origin_index = origins.index
    for h in range(1, horizon + 1):
        chunk = origins[feature_cols].copy()
        chunk["horizon"] = h
        chunk[target] = grouped[target].shift(-h).loc[origin_index].values
        chunk["target_date"] = grouped["date"].shift(-h).loc[origin_index].values

        for col in FUTURE_KNOWN_COLS:
            if col in df.columns:
                chunk[col] = grouped[col].shift(-h).loc[origin_index].values

        chunk["store_id"] = origins["store_id"].values
        chunk["sku_id"] = origins["sku_id"].values
        chunk["origin_date"] = origins["date"].values
        chunks.append(chunk)

    out = pd.concat(chunks, ignore_index=True)
    out = out.dropna(subset=[target])
    if max_target_date is not None:
        out = out[out["target_date"] <= max_target_date]
    return out


def _prepare_datasets(
    df: pd.DataFrame,
    split: TemporalSplit,
    feature_cols: list[str],
    target: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    model_features = feature_cols + ["horizon"]
    train_expanded = _build_direct_frame(
        df, split.mask(df["date"], "train"), feature_cols, target, horizon, max_target_date=split.train_end
    )
    val_expanded = _build_direct_frame(
        df, split.mask(df["date"], "val"), feature_cols, target, horizon, max_target_date=split.val_end
    )
    return (
        train_expanded[model_features],
        train_expanded[target],
        val_expanded[model_features],
        val_expanded[target],
        model_features,
    )


def _apply_future_known_cols(rows: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    for col in FUTURE_KNOWN_COLS:
        future_col = f"{col}_future" if f"{col}_future" in merged.columns else col
        if future_col in merged.columns:
            out[col] = merged[future_col].values
        elif col in merged.columns:
            out[col] = merged[col].values
    return out


def _synthesize_future_known(rows: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Fill future-known inputs from origin values plus calendar for target_date."""
    out = rows.copy()
    weekday = target_date.weekday()
    month = target_date.month
    out["weekday"] = weekday
    out["month"] = month
    out["is_weekend"] = int(weekday >= 5)
    out["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["effective_price"] = out["list_price"] * (1 - out["discount_pct"])
    out["promo_x_weekend"] = out["promo_flag"] * out["is_weekend"]
    if "rain_mm" in out.columns:
        out["rain_heavy"] = (out["rain_mm"] >= out["rain_mm"].median()).astype(int)
    if "temperature" in out.columns:
        out["temp_bucket"] = pd.cut(
            out["temperature"],
            bins=[-np.inf, 5, 15, 25, np.inf],
            labels=[0, 1, 2, 3],
        ).astype(int)
    return out


def _merged_meta_column(merged: pd.DataFrame, col: str):
    future_col = f"{col}_future"
    if future_col in merged.columns:
        return merged[future_col].values
    if col in merged.columns:
        return merged[col].values
    return np.full(len(merged), np.nan)


def predict_forward_tree_model(
    model,
    df: pd.DataFrame,
    origin_date: pd.Timestamp,
    feature_cols: list[str],
    horizon: int,
    target: str = "units_sold",
) -> pd.DataFrame:
    """Production inference from a single origin date.

    Uses observed future rows when present (backtest). Otherwise synthesizes
    future-known promo/calendar/weather inputs from the origin row.
    """
    origin_date = pd.Timestamp(origin_date).normalize()
    origins = df[df["date"] == origin_date].copy()
    if origins.empty:
        return pd.DataFrame()

    test_dates = pd.date_range(
        origin_date + pd.Timedelta(days=1),
        origin_date + pd.Timedelta(days=horizon),
        freq="D",
    )
    model_features = feature_cols + ["horizon"]
    predictions = []

    for h, target_date in enumerate(test_dates, start=1):
        future_cols = [
            "store_id",
            "sku_id",
            target,
            "channel",
            "category",
            "promo_flag",
            "purchase_cost",
            "margin_pct",
        ] + [c for c in FUTURE_KNOWN_COLS if c in df.columns]
        future_cols = list(dict.fromkeys(future_cols))
        future = df[df["date"] == target_date][future_cols]

        if future.empty:
            merged = origins.copy()
        else:
            merged = origins.merge(future, on=["store_id", "sku_id"], suffixes=("", "_future"))
        if merged.empty:
            continue

        rows = merged[feature_cols].copy()
        rows["horizon"] = h
        if future.empty:
            rows = _synthesize_future_known(rows, target_date)
        else:
            rows = _apply_future_known_cols(rows, merged)

        preds = model.predict(rows[model_features])
        predictions.append(
            pd.DataFrame(
                {
                    "store_id": merged["store_id"].to_numpy(),
                    "sku_id": merged["sku_id"].to_numpy(),
                    "date": target_date,
                    "origin_date": origin_date,
                    "horizon": h,
                    "prediction": np.maximum(preds, 0.0),
                    target: _merged_meta_column(merged, target),
                    "channel": _merged_meta_column(merged, "channel"),
                    "category": _merged_meta_column(merged, "category"),
                    "promo_flag": _merged_meta_column(merged, "promo_flag"),
                    "purchase_cost": _merged_meta_column(merged, "purchase_cost"),
                    "margin_pct": _merged_meta_column(merged, "margin_pct"),
                }
            )
        )

    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def predict_test_tree_model(
    model,
    df: pd.DataFrame,
    split: TemporalSplit,
    feature_cols: list[str],
    target: str = "units_sold",
) -> pd.DataFrame:
    origin_date = split.test_start - pd.Timedelta(days=1)
    origins = df[df["date"] == origin_date].copy()
    test_dates = pd.date_range(split.test_start, split.test_end, freq="D")
    model_features = feature_cols + ["horizon"]

    predictions = []
    for h, target_date in enumerate(test_dates, start=1):
        future_cols = [
            "store_id",
            "sku_id",
            target,
            "channel",
            "category",
            "promo_flag",
            "purchase_cost",
            "margin_pct",
        ] + [c for c in FUTURE_KNOWN_COLS if c in df.columns]
        future_cols = list(dict.fromkeys(future_cols))
        future = df[df["date"] == target_date][future_cols]
        merged = origins.merge(future, on=["store_id", "sku_id"], suffixes=("", "_future"))
        if merged.empty:
            continue

        rows = merged[feature_cols].copy()
        rows["horizon"] = h
        rows = _apply_future_known_cols(rows, merged)

        preds = model.predict(rows[model_features])

        predictions.append(
            pd.DataFrame(
                {
                    "store_id": merged["store_id"].to_numpy(),
                    "sku_id": merged["sku_id"].to_numpy(),
                    "date": target_date,
                    "origin_date": origin_date,
                    "horizon": h,
                    "prediction": np.maximum(preds, 0.0),
                    target: _merged_meta_column(merged, target),
                    "channel": _merged_meta_column(merged, "channel"),
                    "category": _merged_meta_column(merged, "category"),
                    "promo_flag": _merged_meta_column(merged, "promo_flag"),
                    "purchase_cost": _merged_meta_column(merged, "purchase_cost"),
                    "margin_pct": _merged_meta_column(merged, "margin_pct"),
                }
            )
        )

    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


class LightGBMForecaster:
    def __init__(self, config: TreeBoostConfig | None = None, horizon: int = 14):
        self.config = config or TreeBoostConfig()
        self.horizon = horizon
        self.model: lgb.Booster | None = None
        self.feature_cols: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        split: TemporalSplit,
        feature_cols: list[str],
        target: str = "units_sold",
    ) -> LightGBMForecaster:
        X_train, y_train, X_val, y_val, self.feature_cols = _prepare_datasets(
            df, split, feature_cols, target, self.horizon
        )
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "verbosity": -1,
            "seed": 42,
        }
        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)],
        )
        return self

    def predict_test(
        self,
        df: pd.DataFrame,
        split: TemporalSplit,
        feature_cols: list[str],
        target: str = "units_sold",
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return predict_test_tree_model(self.model, df, split, feature_cols, target)


class XGBoostForecaster:
    def __init__(self, config: TreeBoostConfig | None = None, horizon: int = 14):
        self.config = config or TreeBoostConfig()
        self.horizon = horizon
        self.model: XGBRegressor | None = None
        self.feature_cols: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        split: TemporalSplit,
        feature_cols: list[str],
        target: str = "units_sold",
    ) -> XGBoostForecaster:
        X_train, y_train, X_val, y_val, self.feature_cols = _prepare_datasets(
            df, split, feature_cols, target, self.horizon
        )
        self.model = XGBRegressor(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=8 if self.config.max_depth == -1 else self.config.max_depth,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            objective="reg:squarederror",
            eval_metric="mae",
            early_stopping_rounds=self.config.early_stopping_rounds,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict_test(
        self,
        df: pd.DataFrame,
        split: TemporalSplit,
        feature_cols: list[str],
        target: str = "units_sold",
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return predict_test_tree_model(self.model, df, split, feature_cols, target)


class CatBoostForecaster:
    def __init__(self, config: TreeBoostConfig | None = None, horizon: int = 14):
        self.config = config or TreeBoostConfig()
        self.horizon = horizon
        self.model: CatBoostRegressor | None = None
        self.feature_cols: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        split: TemporalSplit,
        feature_cols: list[str],
        target: str = "units_sold",
    ) -> CatBoostForecaster:
        X_train, y_train, X_val, y_val, self.feature_cols = _prepare_datasets(
            df, split, feature_cols, target, self.horizon
        )
        self.model = CatBoostRegressor(
            iterations=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            depth=8 if self.config.max_depth == -1 else self.config.max_depth,
            subsample=self.config.subsample,
            loss_function="MAE",
            eval_metric="MAE",
            early_stopping_rounds=self.config.early_stopping_rounds,
            random_seed=42,
            verbose=False,
        )
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        return self

    def predict_test(
        self,
        df: pd.DataFrame,
        split: TemporalSplit,
        feature_cols: list[str],
        target: str = "units_sold",
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return predict_test_tree_model(self.model, df, split, feature_cols, target)
