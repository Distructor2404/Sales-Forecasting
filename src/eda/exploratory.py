"""Exploratory data analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save_fig(fig, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / name, dpi=120)
    plt.close(fig)


def run_eda(df: pd.DataFrame, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    work = df.copy()
    work["month_num"] = work["date"].dt.month
    work["year_num"] = work["date"].dt.year

    summary = {
        "n_rows": len(work),
        "date_min": str(work["date"].min().date()),
        "date_max": str(work["date"].max().date()),
        "n_stores": int(work["store_id"].nunique()),
        "n_skus": int(work["sku_id"].nunique()),
        "n_pairs": int(work.groupby(["store_id", "sku_id"]).ngroups),
        "n_countries": int(work["country"].nunique()),
        "n_cities": int(work["city"].nunique()),
        "stockout_rate": float(work["stock_out_flag"].mean()),
        "promo_rate": float(work["promo_flag"].mean()),
        "holiday_rate": float(work["is_holiday"].mean()),
        "avg_units_sold": float(work["units_sold"].mean()),
        "median_units_sold": float(work["units_sold"].median()),
        "avg_discount_pct": float(work.loc[work["promo_flag"] == 1, "discount_pct"].mean()),
    }

    # --- 1. Total daily demand ---
    daily = work.groupby("date")["units_sold"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["date"], daily["units_sold"], linewidth=0.8, color="#2c3e50")
    ax.set_title("Total Daily Demand (All Stores/SKUs)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    _save_fig(fig, output_dir, "seasonality_total_demand.png")

    # --- 2. Promo lift ---
    promo_lift = (
        work.groupby("promo_flag")["units_sold"]
        .agg(["mean", "median", "count"])
        .rename(index={0: "non_promo", 1: "promo"})
    )
    summary["promo_lift_mean"] = float(
        promo_lift.loc["promo", "mean"] / promo_lift.loc["non_promo", "mean"]
    )
    summary["promo_lift_median"] = float(
        promo_lift.loc["promo", "median"] / promo_lift.loc["non_promo", "median"]
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    promo_lift["mean"].plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Promo vs Non-Promo Average Demand")
    ax.set_ylabel("Mean Units Sold")
    ax.set_xticklabels(["Non-Promo", "Promo"], rotation=0)
    _save_fig(fig, output_dir, "promo_lift.png")

    # --- 3. Promo lift by category ---
    promo_cat = (
        work.groupby(["category", "promo_flag"])["units_sold"]
        .mean()
        .unstack(fill_value=0)
    )
    promo_cat.columns = ["Non-Promo", "Promo"]
    promo_cat["lift"] = promo_cat["Promo"] / promo_cat["Non-Promo"].replace(0, np.nan)
    summary["promo_lift_by_category"] = promo_cat["lift"].dropna().to_dict()

    fig, ax = plt.subplots(figsize=(9, 4))
    promo_cat[["Non-Promo", "Promo"]].plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Promo vs Non-Promo by Category")
    ax.set_ylabel("Mean Units Sold")
    ax.legend(title="")
    _save_fig(fig, output_dir, "promo_lift_by_category.png")

    # --- 4. Stockout effect ---
    stockout_compare = work.groupby("stock_out_flag")["units_sold"].agg(["mean", "count"])
    summary["stockout_demand_ratio"] = float(
        stockout_compare.loc[1, "mean"] / max(stockout_compare.loc[0, "mean"], 1e-6)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    stockout_compare["mean"].plot(kind="bar", ax=ax, color=["#55A868", "#C44E52"])
    ax.set_xticklabels(["Normal", "Stockout"], rotation=0)
    ax.set_title("Average Demand: Normal vs Stockout Days")
    ax.set_ylabel("Mean Units Sold")
    _save_fig(fig, output_dir, "stockout_effect.png")

    # --- 5. Channel & category ---
    channel_stats = work.groupby("channel")["units_sold"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    channel_stats.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title("Average Demand by Channel")
    ax.set_ylabel("Mean Units Sold")
    _save_fig(fig, output_dir, "channel_heterogeneity.png")

    cat_stats = work.groupby("category")["units_sold"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    cat_stats.plot(kind="bar", ax=ax, color="#C44E52")
    ax.set_title("Average Demand by Category")
    ax.set_ylabel("Mean Units Sold")
    _save_fig(fig, output_dir, "category_heterogeneity.png")

    # --- 6. Monthly seasonality ---
    monthly = work.groupby("month_num")["units_sold"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly.plot(kind="line", marker="o", ax=ax, color="#e74c3c")
    ax.set_title("Monthly Seasonality (Average Units Sold)")
    ax.set_xlabel("Month")
    _save_fig(fig, output_dir, "monthly_seasonality.png")

    # --- 7. Weekday pattern ---
    weekday = work.groupby("weekday")["units_sold"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    weekday.plot(kind="bar", ax=ax, color="#3498db")
    ax.set_title("Average Demand by Weekday (0=Mon)")
    ax.set_xlabel("Weekday")
    _save_fig(fig, output_dir, "weekday_seasonality.png")

    # --- 8. Weekend vs weekday ---
    weekend = work.groupby("is_weekend")["units_sold"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    weekend.plot(kind="bar", ax=ax, color=["#9b59b6", "#1abc9c"])
    ax.set_xticklabels(["Weekday", "Weekend"], rotation=0)
    ax.set_title("Weekday vs Weekend Demand")
    ax.set_ylabel("Mean Units Sold")
    _save_fig(fig, output_dir, "weekend_effect.png")

    # --- 9. Holiday effect ---
    holiday = work.groupby("is_holiday")["units_sold"].mean()
    summary["holiday_lift"] = float(holiday.get(1, 0) / max(holiday.get(0, 1), 1e-6))
    fig, ax = plt.subplots(figsize=(5, 4))
    holiday.plot(kind="bar", ax=ax, color=["#95a5a6", "#e67e22"])
    ax.set_xticklabels(["Non-Holiday", "Holiday"], rotation=0)
    ax.set_title("Holiday vs Non-Holiday Demand")
    ax.set_ylabel("Mean Units Sold")
    _save_fig(fig, output_dir, "holiday_effect.png")

    # --- 10. Heatmap: weekday x month ---
    heat = work.pivot_table(index="weekday", columns="month_num", values="units_sold", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat, cmap="YlOrRd", annot=False, ax=ax, cbar_kws={"label": "Mean units sold"})
    ax.set_title("Demand Heatmap: Weekday × Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Weekday")
    _save_fig(fig, output_dir, "weekday_month_heatmap.png")

    # --- 11. Discount vs demand (promo rows) ---
    promo_rows = work[work["promo_flag"] == 1].copy()
    if len(promo_rows) > 0:
        promo_rows["discount_bin"] = pd.cut(promo_rows["discount_pct"], bins=5)
        disc = promo_rows.groupby("discount_bin", observed=True)["units_sold"].mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        disc.plot(kind="bar", ax=ax, color="#DD8452")
        ax.set_title("Average Demand by Discount Level (Promo Days)")
        ax.set_xlabel("Discount % bin")
        ax.tick_params(axis="x", rotation=30)
        _save_fig(fig, output_dir, "discount_vs_demand.png")

    # --- 12. Weather ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sample_weather = work.sample(min(5000, len(work)), random_state=42)
    axes[0].scatter(sample_weather["temperature"], sample_weather["units_sold"], alpha=0.15, s=8)
    axes[0].set_title("Temperature vs Units Sold")
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Units Sold")
    rain_grp = work.groupby(work["rain_mm"] > work["rain_mm"].median())["units_sold"].mean()
    rain_grp.index = ["Dry", "Rainy"]
    rain_grp.plot(kind="bar", ax=axes[1], color=["#f1c40f", "#3498db"])
    axes[1].set_title("Dry vs Rainy Days")
    axes[1].set_ylabel("Mean Units Sold")
    _save_fig(fig, output_dir, "weather_effects.png")

    # --- 13. Units sold distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(work["units_sold"], bins=50, color="#8e44ad", edgecolor="white", alpha=0.85)
    ax.set_title("Distribution of Daily Units Sold")
    ax.set_xlabel("Units Sold")
    ax.set_ylabel("Frequency")
    _save_fig(fig, output_dir, "units_sold_distribution.png")

    # --- 14. Top stores by volume ---
    top_stores = work.groupby("store_id")["units_sold"].sum().sort_values(ascending=False).head(13)
    fig, ax = plt.subplots(figsize=(10, 4))
    top_stores.plot(kind="bar", ax=ax, color="#16a085")
    ax.set_title("Total Units Sold by Store")
    ax.set_ylabel("Total Units")
    _save_fig(fig, output_dir, "store_volume.png")

    # --- 15. Country breakdown ---
    if work["country"].nunique() > 1:
        country = work.groupby("country")["units_sold"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        country.plot(kind="bar", ax=ax, color="#2980b9")
        ax.set_title("Average Demand by Country")
        _save_fig(fig, output_dir, "country_demand.png")

    # Save summary JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
