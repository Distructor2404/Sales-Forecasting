"""End-to-end forecasting pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.features import build_features, get_ml_feature_columns
from src.data.loading import load_data
from src.data.splits import create_temporal_split
from src.eda.exploratory import run_eda
from src.evaluation.metrics import (
    breakdown_metrics,
    business_proxy_metrics,
    compute_metrics,
    save_metrics_report,
)
from src.models.baseline import (
    CatBoostConfig,
    CatBoostForecaster,
    LightGBMConfig,
    LightGBMForecaster,
    XGBoostConfig,
    XGBoostForecaster,
)
from src.models.deep_learning import DeepLearningConfig, DeepLearningForecaster


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str | Path = "config.yaml") -> dict:
    cfg = load_config(config_path)
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(cfg["data"]["path"])
    group_cols = cfg["data"]["group_cols"]
    target = cfg["data"]["target"]
    horizon = cfg["data"]["horizon"]

    split = create_temporal_split(
        df,
        test_days=cfg["splits"]["test_days"],
        val_days=cfg["splits"]["val_days"],
    )
    print(
        f"Split: train<={split.train_end.date()}, "
        f"val={split.val_start.date()}..{split.val_end.date()}, "
        f"test={split.test_start.date()}..{split.test_end.date()}"
    )

    print("Running EDA...")
    eda_summary = run_eda(df, output_dir / "eda")
    with open(output_dir / "eda" / "summary.json", "w") as f:
        json.dump(eda_summary, f, indent=2)

    print("Building features...")
    train_for_encoder = df[split.mask(df["date"], "train")]
    _, encoder = build_features(
        df,
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

    tree_models = [
        ("lightgbm", LightGBMForecaster, LightGBMConfig, cfg["models"]["lightgbm"]),
        ("xgboost", XGBoostForecaster, XGBoostConfig, cfg["models"]["xgboost"]),
        ("catboost", CatBoostForecaster, CatBoostConfig, cfg["models"]["catboost"]),
    ]

    all_predictions: dict[str, pd.DataFrame] = {}

    for name, forecaster_cls, config_cls, model_cfg in tree_models:
        print(f"Training {name}...", flush=True)
        model = forecaster_cls(config=config_cls(**model_cfg), horizon=horizon)
        model.fit(featured_df, split, feature_cols, target=target)
        preds = model.predict_test(featured_df, split, feature_cols, target=target)
        preds.to_csv(output_dir / f"predictions_{name}.csv", index=False)
        all_predictions[name] = preds

    print("Training deep learning model...", flush=True)
    dl_cfg = DeepLearningConfig(
        **cfg["models"]["deep_learning"],
        history_length=cfg["features"]["history_length"],
        horizon=horizon,
    )
    dl_model = DeepLearningForecaster(config=dl_cfg)
    dl_model.fit(featured_df, split)
    dl_preds = dl_model.predict_test()
    dl_preds.to_csv(output_dir / "predictions_deep_learning.csv", index=False)
    all_predictions["deep_learning"] = dl_preds

    print("Evaluating models...")
    results = {}
    comparison_rows = []

    for name, preds in all_predictions.items():
        overall = compute_metrics(preds[target].values, preds["prediction"].values)
        breakdowns = {
            "channel": breakdown_metrics(preds, group_col="channel"),
            "category": breakdown_metrics(preds, group_col="category"),
            "promo": breakdown_metrics(preds, group_col="promo_flag"),
        }
        business = business_proxy_metrics(preds)
        save_metrics_report(
            overall,
            breakdowns,
            output_dir / f"metrics_{name}.json",
            business=business,
        )
        results[name] = {"overall": overall, "business": business}
        comparison_rows.append({"model": name, **overall, **business})

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))

    summary = {
        "split": {
            "train_end": str(split.train_end.date()),
            "val_start": str(split.val_start.date()),
            "val_end": str(split.val_end.date()),
            "test_start": str(split.test_start.date()),
            "test_end": str(split.test_end.date()),
        },
        "eda": eda_summary,
        "results": results,
    }
    with open(output_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPipeline complete. Outputs saved to {output_dir}/")
    return summary


if __name__ == "__main__":
    run_pipeline()
