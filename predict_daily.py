#!/usr/bin/env python3
"""Daily batch inference using saved production artifacts (joblib)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.production.inference import run_batch_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production batch forecast")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory with production_bundle.joblib (default from config)",
    )
    parser.add_argument("--data", default=None, help="Path to data.csv (default from config)")
    parser.add_argument(
        "--origin-date",
        default=None,
        help="Forecast origin date YYYY-MM-DD (default: latest date in data)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default from config)",
    )
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prod_cfg = cfg.get("production", {})
    artifacts_dir = args.artifacts_dir or prod_cfg.get("artifacts_dir", "outputs/models/production")
    data_path = args.data or cfg["data"]["path"]
    output_path = args.output or prod_cfg.get(
        "predictions_path", "outputs/production/predictions_latest.csv"
    )

    origin = pd.Timestamp(args.origin_date) if args.origin_date else None

    preds = run_batch_inference(
        data_path=data_path,
        artifacts_dir=artifacts_dir,
        config_path=args.config,
        origin_date=origin,
        output_path=output_path,
    )

    print(f"Forecasts written: {output_path}")
    print(f"Rows: {len(preds):,}")
    if preds.empty:
        print("No forecasts generated — check origin date and input data.")
        return

    print(f"Origin: {preds['origin_date'].iloc[0].date()}")
    print(f"Horizon dates: {preds['date'].min().date()} → {preds['date'].max().date()}")
    print(preds[["store_id", "sku_id", "date", "horizon", "prediction"]].head())


if __name__ == "__main__":
    main()
