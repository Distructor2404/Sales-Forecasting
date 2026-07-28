"""FMCG Demand Forecasting — Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard.data_loader import (
    MODEL_FILES,
    load_all_predictions,
    load_eda_summary,
    load_metrics,
    load_model_comparison,
    load_predictions,
    load_raw_data,
    list_eda_images,
    output_dir,
)

st.set_page_config(
    page_title="FMCG Demand Forecast",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_OPTIONS = list(MODEL_FILES.keys())


@st.cache_data
def cached_comparison() -> pd.DataFrame:
    return load_model_comparison()


@st.cache_data
def cached_predictions(model: str) -> pd.DataFrame:
    return load_predictions(model)


@st.cache_data
def cached_all_predictions() -> dict[str, pd.DataFrame]:
    return load_all_predictions()


@st.cache_data
def cached_eda() -> dict:
    return load_eda_summary()


def _metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value, help=help_text or None)


def render_overview(comparison: pd.DataFrame, eda: dict) -> None:
    st.header("Model overview")
    st.caption("14-day test forecast · Dec 18–31, 2023 · Origin: Dec 17, 2023")

    if comparison.empty:
        st.warning("Run the pipeline first: `PYTHONPATH=. python run_pipeline.py`")
        return

    best = comparison.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Best model", str(best["model"]), "Lowest WAPE on test set")
    with c2:
        _metric_card("Best WAPE", f"{best['wape']:.1%}", "Weighted Absolute Percentage Error")
    with c3:
        _metric_card("Best MAE", f"{best['mae']:.1f} units", "Mean absolute error per store–SKU–day")
    with c4:
        _metric_card("Forecast rows", "14,056", "~1,004 pairs × 14 days")

    if eda:
        st.subheader("Dataset snapshot")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Rows", f"{eda.get('n_rows', 0):,}")
        d2.metric("Stores", eda.get("n_stores", "—"))
        d3.metric("SKUs", eda.get("n_skus", "—"))
        d4.metric("Store–SKU pairs", eda.get("n_pairs", "—"))
        d5.metric("Promo lift", f"{eda.get('promo_lift_mean', 0):.2f}×")

    st.subheader("Model comparison")
    display = comparison.copy()
    display["wape"] = display["wape"].map(lambda x: f"{x:.1%}")
    display["mae"] = display["mae"].map(lambda x: f"{x:.2f}")
    display["rmse"] = display["rmse"].map(lambda x: f"{x:.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    fig = make_subplots(rows=1, cols=3, subplot_titles=("WAPE ↓", "MAE ↓", "RMSE ↓"))
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(comparison))]
    fig.add_trace(
        go.Bar(x=comparison["model"], y=comparison["wape"], marker_color=colors, name="WAPE"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=comparison["model"], y=comparison["mae"], marker_color=colors, showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=comparison["model"], y=comparison["rmse"], marker_color=colors, showlegend=False),
        row=1,
        col=3,
    )
    fig.update_layout(height=380, showlegend=False, margin=dict(t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_explorer(df: pd.DataFrame, model: str) -> None:
    st.header("Forecast explorer")
    st.caption(f"Model: **{model}** — actual vs predicted for a single store–SKU series")

    stores = sorted(df["store_id"].unique())
    c1, c2 = st.columns(2)
    with c1:
        store = st.selectbox("Store", stores, key="explorer_store")
    skus = sorted(df[df["store_id"] == store]["sku_id"].unique())
    with c2:
        sku = st.selectbox("SKU", skus, key="explorer_sku")

    series = df[(df["store_id"] == store) & (df["sku_id"] == sku)].sort_values("date")
    if series.empty:
        st.info("No data for this selection.")
        return

    meta = series.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Channel", meta["channel"])
    m2.metric("Category", meta["category"])
    m3.metric("MAE (14 days)", f"{series['abs_error'].mean():.1f}")
    m4.metric("Total actual", f"{series['units_sold'].sum():,.0f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["units_sold"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#e74c3c", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["prediction"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color="#2ecc71", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"{store} × {sku} — 14-day forecast",
        xaxis_title="Date",
        yaxis_title="Units sold",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    detail = series[
        ["date", "horizon", "units_sold", "prediction", "error", "promo_flag", "purchase_cost"]
    ].copy()
    detail["units_sold"] = detail["units_sold"].round(0).astype(int)
    detail["prediction"] = detail["prediction"].round(1)
    detail["error"] = detail["error"].round(1)
    st.dataframe(detail, use_container_width=True, hide_index=True)


def render_aggregate(df: pd.DataFrame, model: str) -> None:
    st.header("Aggregate forecasts")
    st.caption(f"Model: **{model}** — rolled up across all store–SKU pairs")

    daily = (
        df.groupby("date", as_index=False)
        .agg(units_sold=("units_sold", "sum"), prediction=("prediction", "sum"))
        .sort_values("date")
    )
    daily["error"] = daily["prediction"] - daily["units_sold"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["date"], y=daily["units_sold"], name="Actual", marker_color="#e74c3c"))
    fig.add_trace(go.Bar(x=daily["date"], y=daily["prediction"], name="Predicted", marker_color="#3498db"))
    fig.update_layout(
        barmode="group",
        title="Total daily demand — all pairs",
        xaxis_title="Date",
        yaxis_title="Units",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        by_channel = (
            df.groupby("channel", as_index=False)
            .agg(units_sold=("units_sold", "sum"), prediction=("prediction", "sum"))
        )
        by_channel["wape"] = (by_channel["prediction"] - by_channel["units_sold"]).abs() / by_channel["units_sold"]
        fig_ch = px.bar(
            by_channel,
            x="channel",
            y=["units_sold", "prediction"],
            barmode="group",
            title="By channel",
            color_discrete_sequence=["#e74c3c", "#3498db"],
        )
        fig_ch.update_layout(height=360)
        st.plotly_chart(fig_ch, use_container_width=True)

    with col_b:
        by_cat = (
            df.groupby("category", as_index=False)
            .agg(units_sold=("units_sold", "sum"), prediction=("prediction", "sum"))
        )
        fig_cat = px.bar(
            by_cat,
            x="category",
            y=["units_sold", "prediction"],
            barmode="group",
            title="By category",
            color_discrete_sequence=["#e74c3c", "#3498db"],
        )
        fig_cat.update_layout(height=360)
        st.plotly_chart(fig_cat, use_container_width=True)


def render_error_analysis(df: pd.DataFrame, model: str, metrics: dict | None) -> None:
    st.header("Error analysis")
    st.caption(f"Model: **{model}**")

    c1, c2 = st.columns(2)
    with c1:
        by_horizon = (
            df.groupby("horizon", as_index=False)
            .agg(mae=("abs_error", "mean"), rmse=("error", lambda s: (s**2).mean() ** 0.5))
        )
        fig_h = px.line(
            by_horizon,
            x="horizon",
            y="mae",
            markers=True,
            title="MAE by forecast horizon (day 1 → 14)",
            labels={"horizon": "Days ahead", "mae": "MAE (units)"},
        )
        fig_h.update_layout(height=360)
        st.plotly_chart(fig_h, use_container_width=True)

    with c2:
        df_promo = df.copy()
        df_promo["promo_label"] = df_promo["promo_flag"].map({0: "Non-promo", 1: "Promo"})
        by_promo = df_promo.groupby("promo_label", as_index=False).agg(mae=("abs_error", "mean"))
        fig_p = px.bar(by_promo, x="promo_label", y="mae", title="MAE: promo vs non-promo", color="promo_label")
        fig_p.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)

    if metrics and "breakdowns" in metrics:
        st.subheader("Metric breakdowns")
        bc1, bc2, bc3 = st.columns(3)
        for col, key, container in [
            (bc1, "channel", bc1),
            (bc2, "category", bc2),
            (bc3, "promo", bc3),
        ]:
            with container:
                rows = [r for r in metrics["breakdowns"][key] if r["group"] != "overall"]
                if rows:
                    bdf = pd.DataFrame(rows)[["group", "wape", "mae"]]
                    bdf["wape"] = bdf["wape"].map(lambda x: f"{x:.1%}")
                    bdf["mae"] = bdf["mae"].map(lambda x: f"{x:.1f}")
                    st.markdown(f"**By {key}**")
                    st.dataframe(bdf, use_container_width=True, hide_index=True)

    st.subheader("Error distribution")
    fig_hist = px.histogram(
        df,
        x="error",
        nbins=50,
        title="Forecast error (prediction − actual)",
        labels={"error": "Error (units)"},
        color_discrete_sequence=["#9b59b6"],
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="black")
    fig_hist.update_layout(height=320)
    st.plotly_chart(fig_hist, use_container_width=True)


def render_model_compare(all_preds: dict[str, pd.DataFrame]) -> None:
    st.header("Compare models")
    if len(all_preds) < 2:
        st.info("Need at least two prediction files to compare models.")
        return

    first_df = next(iter(all_preds.values()))
    stores = sorted(first_df["store_id"].unique())
    c1, c2 = st.columns(2)
    with c1:
        store = st.selectbox("Store", stores, key="compare_store")
    skus = sorted(first_df[first_df["store_id"] == store]["sku_id"].unique())
    with c2:
        sku = st.selectbox("SKU", skus, key="compare_sku")

    actual = first_df[(first_df["store_id"] == store) & (first_df["sku_id"] == sku)].sort_values("date")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual["date"],
            y=actual["units_sold"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="black", width=3),
        )
    )
    palette = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22"]
    for i, (name, pdf) in enumerate(all_preds.items()):
        sub = pdf[(pdf["store_id"] == store) & (pdf["sku_id"] == sku)].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["prediction"],
                mode="lines+markers",
                name=name,
                line=dict(color=palette[i % len(palette)], dash="dash"),
            )
        )
    fig.update_layout(
        title=f"All models — {store} × {sku}",
        xaxis_title="Date",
        yaxis_title="Units sold",
        height=440,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_data_table(df: pd.DataFrame, model: str) -> None:
    st.header("Prediction table")
    st.caption(f"Model: **{model}** — filter and download")

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        ch_filter = st.multiselect("Channel", sorted(df["channel"].unique()), default=[])
    with fc2:
        cat_filter = st.multiselect("Category", sorted(df["category"].unique()), default=[])
    with fc3:
        promo_filter = st.selectbox("Promo", ["All", "Promo only", "Non-promo only"])
    with fc4:
        hz = st.slider("Horizon (days ahead)", 1, 14, (1, 14))

    filtered = df.copy()
    if ch_filter:
        filtered = filtered[filtered["channel"].isin(ch_filter)]
    if cat_filter:
        filtered = filtered[filtered["category"].isin(cat_filter)]
    if promo_filter == "Promo only":
        filtered = filtered[filtered["promo_flag"] == 1]
    elif promo_filter == "Non-promo only":
        filtered = filtered[filtered["promo_flag"] == 0]
    filtered = filtered[(filtered["horizon"] >= hz[0]) & (filtered["horizon"] <= hz[1])]

    st.write(f"**{len(filtered):,}** rows")
    show = filtered[
        [
            "store_id",
            "sku_id",
            "date",
            "horizon",
            "units_sold",
            "prediction",
            "error",
            "channel",
            "category",
            "promo_flag",
        ]
    ].copy()
    show["prediction"] = show["prediction"].round(1)
    show["error"] = show["error"].round(1)
    st.dataframe(show, use_container_width=True, height=400)

    st.download_button(
        "Download CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{model.lower().replace(' ', '_')}_filtered.csv",
        mime="text/csv",
    )


@st.cache_data
def cached_raw_data() -> pd.DataFrame:
    return load_raw_data()


def render_eda(eda: dict) -> None:
    st.header("Exploratory Data Analysis")
    st.caption("Dataset patterns that drive feature engineering and model design")

    raw = cached_raw_data()
    images = list_eda_images()

    # --- Summary KPIs ---
    if eda:
        st.subheader("Key findings")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Rows", f"{eda.get('n_rows', 0):,}")
        k2.metric("Promo lift", f"{eda.get('promo_lift_mean', 0):.2f}×")
        k3.metric("Stockout rate", f"{eda.get('stockout_rate', 0):.1%}")
        k4.metric("Promo rate", f"{eda.get('promo_rate', 0):.1%}")
        k5.metric("Holiday lift", f"{eda.get('holiday_lift', 0):.2f}×")
        k6.metric(
            "Stockout demand",
            f"{eda.get('stockout_demand_ratio', 0):.0%}",
            help="Stockout-day sales as % of normal days",
        )

        with st.expander("Full EDA summary (JSON)"):
            st.json(eda)

    if raw.empty and not images:
        st.info("Run the pipeline to generate EDA: `PYTHONPATH=. python run_pipeline.py`")
        return

    # --- Interactive charts from raw data ---
    if not raw.empty:
        st.subheader("Interactive exploration")

        ic1, ic2 = st.columns(2)
        with ic1:
            ch = st.selectbox("Filter channel", ["All"] + sorted(raw["channel"].unique()), key="eda_ch")
        with ic2:
            cat = st.selectbox("Filter category", ["All"] + sorted(raw["category"].unique()), key="eda_cat")

        filt = raw.copy()
        if ch != "All":
            filt = filt[filt["channel"] == ch]
        if cat != "All":
            filt = filt[filt["category"] == cat]

        r1, r2 = st.columns(2)
        with r1:
            daily = filt.groupby("date")["units_sold"].sum().reset_index()
            fig = px.line(daily, x="date", y="units_sold", title="Total daily demand")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            promo_df = (
                filt.groupby("promo_flag")["units_sold"]
                .mean()
                .reset_index()
                .assign(label=lambda d: d["promo_flag"].map({0: "Non-promo", 1: "Promo"}))
            )
            fig = px.bar(promo_df, x="label", y="units_sold", title="Promo vs non-promo", color="label")
            fig.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        r3, r4 = st.columns(2)
        with r3:
            wd = filt.groupby("weekday")["units_sold"].mean().reset_index()
            fig = px.bar(wd, x="weekday", y="units_sold", title="By weekday (0=Mon)")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        with r4:
            mo = filt.groupby(filt["date"].dt.month)["units_sold"].mean().reset_index()
            mo.columns = ["month", "units_sold"]
            fig = px.line(mo, x="month", y="units_sold", markers=True, title="Monthly seasonality")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        r5, r6 = st.columns(2)
        with r5:
            stock = (
                filt.groupby("stock_out_flag")["units_sold"]
                .mean()
                .reset_index()
                .assign(label=lambda d: d["stock_out_flag"].map({0: "Normal", 1: "Stockout"}))
            )
            fig = px.bar(stock, x="label", y="units_sold", title="Stockout effect", color="label")
            fig.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with r6:
            if filt["promo_flag"].sum() > 0:
                promo_only = filt[filt["promo_flag"] == 1].copy()
                promo_only["disc_bin"] = pd.cut(promo_only["discount_pct"], bins=5)
                disc = promo_only.groupby("disc_bin", observed=True)["units_sold"].mean().reset_index()
                disc["disc_bin"] = disc["disc_bin"].astype(str)
                fig = px.bar(disc, x="disc_bin", y="units_sold", title="Demand by discount (promo days)")
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        heat = filt.pivot_table(index="weekday", columns=filt["date"].dt.month, values="units_sold", aggfunc="mean")
        if not heat.empty:
            fig = px.imshow(heat, labels=dict(x="Month", y="Weekday", color="Units"), title="Weekday × Month heatmap")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

        # Top SKUs for selected filters
        top = filt.groupby("sku_id")["units_sold"].sum().sort_values(ascending=False).head(15).reset_index()
        fig = px.bar(top, x="sku_id", y="units_sold", title="Top 15 SKUs by total volume")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    # --- Saved static charts from pipeline ---
    if images:
        st.subheader("Saved EDA charts (from pipeline)")
        st.caption(f"Files in `{output_dir() / 'eda'}`")

        chart_groups = {
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

        img_map = {p.stem: p for p in images}
        for group_name, stems in chart_groups.items():
            group_imgs = [img_map[s] for s in stems if s in img_map]
            if not group_imgs:
                continue
            st.markdown(f"**{group_name}**")
            cols = st.columns(2)
            for i, img_path in enumerate(group_imgs):
                with cols[i % 2]:
                    st.image(
                        str(img_path),
                        caption=img_path.stem.replace("_", " ").title(),
                        use_container_width=True,
                    )

        # Any remaining images not in groups
        grouped = {s for stems in chart_groups.values() for s in stems}
        extra = [p for p in images if p.stem not in grouped]
        if extra:
            st.markdown("**Other**")
            cols = st.columns(2)
            for i, img_path in enumerate(extra):
                with cols[i % 2]:
                    st.image(str(img_path), caption=img_path.stem.replace("_", " ").title(), use_container_width=True)


def main() -> None:
    st.title("📦 FMCG Demand Forecasting Dashboard")
    st.markdown(
        "Explore **14-day store–SKU forecasts** from XGBoost, LightGBM, CatBoost, and GRU. "
        f"Data loaded from `{output_dir()}`."
    )

    comparison = cached_comparison()
    eda = cached_eda()
    all_preds = cached_all_predictions()

    if not all_preds:
        st.error(
            "No prediction files found in `outputs/`. "
            "Run: `PYTHONPATH=. python run_pipeline.py`"
        )
        st.stop()

    with st.sidebar:
        st.header("Settings")
        model = st.selectbox("Primary model", MODEL_OPTIONS, index=0)
        try:
            df = cached_predictions(model)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        st.divider()
        st.markdown("**Test window**")
        st.text("2023-12-18 → 2023-12-31")
        st.text("Origin: 2023-12-17")
        st.divider()
        metrics = load_metrics(model)
        if metrics:
            st.markdown("**Selected model metrics**")
            st.text(f"WAPE: {metrics['overall']['wape']:.1%}")
            st.text(f"MAE:  {metrics['overall']['mae']:.1f}")
            st.text(f"RMSE: {metrics['overall']['rmse']:.1f}")

    tab_overview, tab_explorer, tab_agg, tab_error, tab_compare, tab_table, tab_eda = st.tabs(
        [
            "Overview",
            "Forecast explorer",
            "Aggregate",
            "Error analysis",
            "Compare models",
            "Data table",
            "EDA",
        ]
    )

    with tab_overview:
        render_overview(comparison, eda)
    with tab_explorer:
        render_forecast_explorer(df, model)
    with tab_agg:
        render_aggregate(df, model)
    with tab_error:
        render_error_analysis(df, model, metrics)
    with tab_compare:
        render_model_compare(all_preds)
    with tab_table:
        render_data_table(df, model)
    with tab_eda:
        render_eda(eda)


if __name__ == "__main__":
    main()
