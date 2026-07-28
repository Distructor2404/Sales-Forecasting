"""Train and save the production model bundle (joblib)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data.features import build_features, get_ml_feature_columns
from src.data.loading import load_data
from src.data.splits import create_temporal_split
from src.models.baseline import XGBoostConfig, XGBoostForecaster
from src.production.artifacts import ProductionArtifacts


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_production_bundle(
    config_path: str | Path = "config.yaml",
    model_name: str = "xgboost",
) -> ProductionArtifacts:
    cfg = load_config(config_path)
    prod_cfg = cfg.get("production", {})
    artifacts_dir = Path(prod_cfg.get("artifacts_dir", "outputs/models/production"))

    df = load_data(cfg["data"]["path"])
    group_cols = cfg["data"]["group_cols"]
    target = cfg["data"]["target"]
    horizon = cfg["data"]["horizon"]

    split = create_temporal_split(
        df,
        test_days=cfg["splits"]["test_days"],
        val_days=cfg["splits"]["val_days"],
    )

    # Production train window: all rows up to val_end (train + validation)
    prod_train_end = split.val_end
    prod_train_df = df[df["date"] <= prod_train_end]

    print(f"Production training on data <= {prod_train_end.date()} ({len(prod_train_df):,} rows)")

    _, encoder = build_features(
        prod_train_df,
        group_cols=group_cols,
        lag_days=cfg["features"]["lag_days"],
        rolling_windows=cfg["features"]["rolling_windows"],
        fit_encoder=True,
    )
    featured_df, _ = build_features(
        df,
        group_cols=group_cols,
        lag_days=cfg["features"]["lag_days"],
        rolling_windows=cfg["features"]["rolling_windows"],
        encoder=encoder,
    )
    feature_cols = get_ml_feature_columns(encoder)

    # Re-use validation split boundaries for early stopping inside prod_train period
    model_cfg = cfg["models"].get(model_name, cfg["models"]["xgboost"])
    forecaster = XGBoostForecaster(config=XGBoostConfig(**model_cfg), horizon=horizon)
    forecaster.fit(featured_df, split, feature_cols, target=target)

    artifacts = ProductionArtifacts(
        model=forecaster.model,
        model_name=model_name,
        encoder=encoder,
        feature_cols=feature_cols,
        model_feature_cols=forecaster.feature_cols,
        horizon=horizon,
        target=target,
        group_cols=group_cols,
        train_data_end=str(prod_train_end.date()),
        metadata={
            "config_path": str(config_path),
            "n_features": len(feature_cols),
            "description": "Production bundle for 14-day store-SKU demand forecasting",
        },
    )

    bundle_path = artifacts.save(artifacts_dir)
    print(f"Saved production bundle: {bundle_path}")
    print(f"Metadata: {artifacts_dir / 'production_metadata.json'}")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save production model (joblib)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", default="xgboost", choices=["xgboost"], help="Production model type")
    args = parser.parse_args()
    train_production_bundle(config_path=args.config, model_name=args.model)


if __name__ == "__main__":
    main()
