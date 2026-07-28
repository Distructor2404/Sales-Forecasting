"""Evaluation metrics and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return {"wape": wape(y_true, y_pred), "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


def breakdown_metrics(
    eval_df: pd.DataFrame,
    y_true_col: str = "units_sold",
    y_pred_col: str = "prediction",
    group_col: str | None = None,
) -> pd.DataFrame:
    rows = []
    groups = [("overall", eval_df)]
    if group_col:
        for name, grp in eval_df.groupby(group_col):
            groups.append((str(name), grp))

    for name, grp in groups:
        metrics = compute_metrics(grp[y_true_col].values, grp[y_pred_col].values)
        metrics["group"] = name
        metrics["n"] = len(grp)
        rows.append(metrics)

    return pd.DataFrame(rows)[["group", "n", "wape", "mae", "rmse"]]


def business_proxy_metrics(eval_df: pd.DataFrame) -> dict[str, float]:
    """Stockout risk and overstock cost proxies."""
    df = eval_df.copy()
    df["forecast_error"] = df["prediction"] - df["units_sold"]
    df["underforecast"] = np.maximum(-df["forecast_error"], 0)
    df["overforecast"] = np.maximum(df["forecast_error"], 0)

    stockout_risk = float((df["underforecast"] * df.get("purchase_cost", 1)).sum())
    overstock_cost = float((df["overforecast"] * df.get("margin_pct", 0)).sum())
    return {"stockout_risk_proxy": stockout_risk, "overstock_cost_proxy": overstock_cost}


def save_metrics_report(
    overall: dict,
    breakdowns: dict[str, pd.DataFrame],
    output_path: str | Path,
    business: dict | None = None,
) -> None:
    report = {"overall": overall, "breakdowns": {}, "business_proxy": business or {}}
    for name, df in breakdowns.items():
        report["breakdowns"][name] = df.to_dict(orient="records")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
