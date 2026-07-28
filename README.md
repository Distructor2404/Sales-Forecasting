# FMCG Multi-Store Demand Forecasting

Production-oriented pipeline for predicting **14-day daily `units_sold`** for every active `(store_id, sku_id)` pair.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# macOS: brew install libomp  (required for LightGBM)
```

## Run

```bash
PYTHONPATH=. python run_pipeline.py
```

## Dashboard

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Opens a local dashboard to explore forecasts, compare models, and download filtered predictions.

### Streamlit Community Cloud

The live app reads **pre-generated files** from `outputs/` (not `data.csv`). Those files must be **committed to Git** — they are not created on Cloud.

1. Run the pipeline locally once (with `data.csv` on your machine):
   ```bash
   PYTHONPATH=. python run_pipeline.py
   ```
2. Commit dashboard artifacts (~6 MB total):
   ```bash
   git add outputs/predictions_*.csv outputs/model_comparison.csv outputs/metrics_*.json outputs/eda/
   git commit -m "Add pre-generated outputs for Streamlit Cloud dashboard"
   git push
   ```
3. In [share.streamlit.io](https://share.streamlit.io) app settings → **Advanced settings** → set **Requirements file** to `requirements-streamlit.txt` (faster deploy; no PyTorch/LightGBM needed for the dashboard).

If you see *"No prediction files found in outputs/"*, the repo is missing the CSV/JSON/PNG files above — push them and redeploy.

## Notebook

```bash
jupyter notebook notebooks/eda_and_model_prep.ipynb
```

Step-by-step EDA and feature-engineering walkthrough. **Self-contained** — no repo imports; only needs `data.csv` and pip packages.

## Production deployment (joblib)

Train and save the production model bundle:

```bash
PYTHONPATH=. python train_production.py
```

Run daily batch inference (loads saved joblib artifact):

```bash
PYTHONPATH=. python predict_daily.py
```

Artifacts saved to:
- `outputs/models/production/production_bundle.joblib`
- `outputs/models/production/production_metadata.json`

Latest forecasts:
- `outputs/production/predictions_latest.csv`

Outputs are written to `outputs/`:
- `eda/` — exploratory plots and summary
- `predictions_lightgbm.csv` / `predictions_deep_learning.csv`
- `metrics_*.json` — WAPE, MAE, RMSE with channel/category/promo breakdowns
- `model_comparison.csv` — side-by-side model comparison

## Project Structure

```
src/
  data/          # loading, leakage-aware features, temporal splits
  eda/           # seasonality, promo lift, heterogeneity plots
  models/        # XGBoost, LightGBM, CatBoost baselines + multi-series GRU forecaster
  evaluation/    # metrics and business proxies
  pipeline.py    # end-to-end orchestration
config.yaml      # reproducible hyperparameters and split settings
WRITEUP.md       # approach, assumptions, results, next steps
```

## Temporal Split

| Split | Dates |
|-------|-------|
| Train | 2021-01-01 → 2023-11-19 |
| Validation | 2023-11-20 → 2023-12-17 (28 days) |
| Test | 2023-12-18 → 2023-12-31 (14-day horizon) |

## Key Design Choices

- **Leakage control**: lag/rolling features use `shift(1+)` within each store–SKU series; encoders fit on train only.
- **Stockouts**: `stock_out_flag`, `days_since_stockout`, and censored-demand handling documented in features module.
- **LightGBM / XGBoost / CatBoost**: direct multi-horizon regression with known future promo/calendar/weather inputs.
- **Production model**: XGBoost saved via joblib (`outputs/models/production/`).
- **Deep learning**: shared-weight GRU encoder–decoder with static store/channel/category embeddings and 28-day history.

## Results (Test Set)

| Model | WAPE | MAE | RMSE |
|-------|------|-----|------|
| **XGBoost** | **0.267** | **16.65** | **25.59** |
| CatBoost | 0.267 | 16.67 | 25.57 |
| LightGBM | 0.267 | 16.69 | 25.57 |
| GRU (DL) | 0.285 | 17.79 | 27.67 |

All three tree models are within ~0.03% WAPE; **XGBoost** is shipped in production.

See `WRITEUP.md` for full analysis and production recommendations.

**Interview prep:** See [`INTERVIEW_README.md`](INTERVIEW_README.md) for the full flow, topics to study, and Q&A in simple language.

**End-to-end feature & prediction flow:** See [`PIPELINE_FLOW.md`](PIPELINE_FLOW.md) for every feature used and how forecasts are generated.

**30–45 min deep-dive practice:** See [`DEEP_DIVE_SCRIPT.md`](DEEP_DIVE_SCRIPT.md) for a timed mock interview script (Part E: architecture, leakage, failure cases, production).
