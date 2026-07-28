"""Load data for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

MODEL_FILES = {
    "XGBoost": "predictions_xgboost.csv",
    "LightGBM": "predictions_lightgbm.csv",
    "CatBoost": "predictions_catboost.csv",
    "Deep Learning (GRU)": "predictions_deep_learning.csv",
}

METRIC_KEY = {
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "CatBoost": "catboost",
    "Deep Learning (GRU)": "deep_learning",
}


def output_dir() -> Path:
    return OUTPUT_DIR


def load_model_comparison() -> pd.DataFrame:
    path = OUTPUT_DIR / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["model"] = df["model"].str.replace("_", " ").str.title()
    df.loc[df["model"] == "Deep Learning", "model"] = "Deep Learning (GRU)"
    return df


def load_predictions(model_label: str) -> pd.DataFrame:
    filename = MODEL_FILES.get(model_label)
    if not filename:
        raise ValueError(f"Unknown model: {model_label}")
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    df = pd.read_csv(path, parse_dates=["date", "origin_date"])
    df["error"] = df["prediction"] - df["units_sold"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = (df["abs_error"] / df["units_sold"].replace(0, pd.NA)).fillna(0)
    return df


def load_all_predictions() -> dict[str, pd.DataFrame]:
    data = {}
    for label, filename in MODEL_FILES.items():
        path = OUTPUT_DIR / filename
        if path.exists():
            data[label] = load_predictions(label)
    return data


def load_metrics(model_label: str) -> dict | None:
    key = METRIC_KEY.get(model_label)
    if not key:
        return None
    path = OUTPUT_DIR / f"metrics_{key}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_eda_summary() -> dict:
    path = OUTPUT_DIR / "eda" / "summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def raw_data_path() -> Path:
    return OUTPUT_DIR.parent / "data.csv"


def raw_data_available() -> bool:
    return raw_data_path().exists()


def load_raw_data() -> pd.DataFrame:
    """Load full data.csv for interactive EDA in dashboard."""
    if not raw_data_available():
        return pd.DataFrame()
    df = pd.read_csv(raw_data_path(), parse_dates=["date"])
    return df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)


EDA_CHART_GROUPS: dict[str, list[str]] = {
    "Seasonality": [
        "seasonality_total_demand",
        "monthly_seasonality",
        "weekday_seasonality",
        "weekday_month_heatmap",
        "weekend_effect",
        "holiday_effect",
    ],
    "Promo & pricing": ["promo_lift", "promo_lift_by_category", "discount_vs_demand"],
    "Stockouts": ["stockout_effect"],
    "Heterogeneity": ["channel_heterogeneity", "category_heterogeneity", "store_volume", "country_demand"],
    "Weather & distribution": ["weather_effects", "units_sold_distribution"],
}


def list_eda_images() -> list[Path]:
    eda_dir = OUTPUT_DIR / "eda"
    if not eda_dir.exists():
        return []
    return sorted(eda_dir.glob("*.png"))
