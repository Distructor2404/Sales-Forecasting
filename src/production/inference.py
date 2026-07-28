"""Production batch inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.features import build_features
from src.data.loading import load_data
from src.models.baseline import predict_forward_tree_model
from src.production.artifacts import ProductionArtifacts


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_featured_dataframe(df: pd.DataFrame, artifacts: ProductionArtifacts, cfg: dict) -> pd.DataFrame:
    featured_df, _ = build_features(
        df,
        group_cols=artifacts.group_cols,
        lag_days=cfg["features"]["lag_days"],
        rolling_windows=cfg["features"]["rolling_windows"],
        encoder=artifacts.encoder,
    )
    return featured_df


def resolve_origin_date(df: pd.DataFrame, origin_date: pd.Timestamp | None = None) -> pd.Timestamp:
    if origin_date is not None:
        return pd.Timestamp(origin_date).normalize()
    return pd.Timestamp(df["date"].max()).normalize()


def predict_from_origin(
    artifacts: ProductionArtifacts,
    featured_df: pd.DataFrame,
    origin_date: pd.Timestamp,
    target: str | None = None,
) -> pd.DataFrame:
    """Generate horizon-day forecasts from a single origin date."""
    target = target or artifacts.target
    origin_date = pd.Timestamp(origin_date).normalize()

    preds = predict_forward_tree_model(
        artifacts.model,
        featured_df,
        origin_date,
        artifacts.feature_cols,
        horizon=artifacts.horizon,
        target=target,
    )
    preds["model"] = artifacts.model_name
    return preds


def run_batch_inference(
    data_path: str | Path,
    artifacts_dir: str | Path,
    config_path: str | Path = "config.yaml",
    origin_date: pd.Timestamp | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    artifacts = ProductionArtifacts.load(artifacts_dir)
    df = load_data(data_path)
    featured_df = build_featured_dataframe(df, artifacts, cfg)
    origin = resolve_origin_date(df, origin_date)

    preds = predict_from_origin(artifacts, featured_df, origin)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(output_path, index=False)

    return preds
