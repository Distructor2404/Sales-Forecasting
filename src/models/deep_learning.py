"""Multi-series GRU encoder-decoder for 14-day demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.splits import TemporalSplit


@dataclass
class DeepLearningConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 256
    epochs: int = 15
    learning_rate: float = 0.001
    patience: int = 5
    history_length: int = 28
    horizon: int = 14


STATIC_COLS = ["store_id_enc", "channel_enc", "category_enc"]
KNOWN_FUTURE_COLS = [
    "promo_flag",
    "discount_pct",
    "is_weekend",
    "is_holiday",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "temperature",
    "rain_mm",
]
HISTORY_COLS = ["units_sold", "promo_flag", "discount_pct", "stock_out_flag", "temperature", "rain_mm"]


class SeriesDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "history": torch.tensor(s["history"], dtype=torch.float32),
            "static": torch.tensor(s["static"], dtype=torch.float32),
            "future_known": torch.tensor(s["future_known"], dtype=torch.float32),
            "target": torch.tensor(s["target"], dtype=torch.float32),
        }


class GRUForecaster(nn.Module):
    def __init__(
        self,
        history_dim: int,
        static_dim: int,
        future_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(
            input_size=history_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_proj = nn.Linear(static_dim, hidden_size)
        self.future_proj = nn.Linear(future_dim, hidden_size)
        self.decoder_cell = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        history: torch.Tensor,
        static: torch.Tensor,
        future_known: torch.Tensor,
    ) -> torch.Tensor:
        _, h_n = self.encoder(history)
        h = h_n[-1]
        static_emb = self.static_proj(static)

        outputs = []
        dec_h = h + static_emb
        for t in range(self.horizon):
            future_emb = self.future_proj(future_known[:, t, :])
            dec_h = self.decoder_cell(future_emb, dec_h)
            combined = torch.cat([dec_h, h], dim=-1)
            outputs.append(self.output_head(combined))

        return torch.cat(outputs, dim=-1).squeeze(-1)


def _build_samples(
    df: pd.DataFrame,
    split_mask: pd.Series,
    history_length: int,
    horizon: int,
    sample_stride: int = 7,
) -> list[dict]:
    """Build sequence samples at origin dates matching split_mask (optionally strided)."""
    samples = []
    group_cols = ["store_id", "sku_id"]
    masked = df.loc[split_mask, group_cols + ["date"]]

    for (store, sku), origin_rows in masked.groupby(group_cols, sort=False):
        grp = df[(df["store_id"] == store) & (df["sku_id"] == sku)].sort_values("date").reset_index(drop=True)
        if len(grp) < history_length + horizon:
            continue

        origin_dates = set(origin_rows["date"])
        for i in range(history_length - 1, len(grp) - horizon):
            if sample_stride > 1 and i % sample_stride != 0:
                continue
            if grp.loc[i, "date"] not in origin_dates:
                continue

            start = i - history_length + 1
            window = grp.iloc[start : i + horizon + 1]
            history = window.iloc[:history_length][HISTORY_COLS].values.astype(np.float32)
            static = window.iloc[history_length - 1][STATIC_COLS].values.astype(np.float32)
            future_known = window.iloc[history_length : history_length + horizon][KNOWN_FUTURE_COLS].values.astype(np.float32)
            target = window.iloc[history_length : history_length + horizon]["units_sold"].values.astype(np.float32)

            samples.append(
                {
                    "history": history,
                    "static": static,
                    "future_known": future_known,
                    "target": target,
                    "store_id": store,
                    "sku_id": sku,
                    "origin_date": grp.loc[i, "date"],
                    "target_dates": window.iloc[history_length : history_length + horizon]["date"].tolist(),
                    "channel": window.iloc[history_length]["channel"],
                    "category": window.iloc[history_length]["category"],
                    "promo_flags": window.iloc[history_length : history_length + horizon]["promo_flag"].tolist(),
                    "purchase_cost": window.iloc[history_length : history_length + horizon]["purchase_cost"].tolist(),
                    "margin_pct": window.iloc[history_length : history_length + horizon]["margin_pct"].tolist(),
                }
            )
    return samples


class DeepLearningForecaster:
    def __init__(self, config: DeepLearningConfig | None = None):
        self.config = config or DeepLearningConfig()
        self.model: GRUForecaster | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, df: pd.DataFrame, split: TemporalSplit) -> DeepLearningForecaster:
        cfg = self.config
        origin_mask = df["date"] <= split.train_end
        val_origin_mask = (df["date"] >= split.val_start) & (df["date"] <= split.val_end)

        train_samples = _build_samples(df, origin_mask, cfg.history_length, cfg.horizon, sample_stride=7)
        val_samples = _build_samples(df, val_origin_mask, cfg.history_length, cfg.horizon, sample_stride=1)

        self.model = GRUForecaster(
            history_dim=len(HISTORY_COLS),
            static_dim=len(STATIC_COLS),
            future_dim=len(KNOWN_FUTURE_COLS),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            horizon=cfg.horizon,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        criterion = nn.L1Loss()

        train_loader = DataLoader(SeriesDataset(train_samples), batch_size=cfg.batch_size, shuffle=False)
        val_loader = DataLoader(SeriesDataset(val_samples), batch_size=cfg.batch_size, shuffle=False)

        best_val = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(cfg.epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                pred = self.model(
                    batch["history"].to(self.device),
                    batch["static"].to(self.device),
                    batch["future_known"].to(self.device),
                )
                loss = criterion(pred, batch["target"].to(self.device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch["target"])

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    pred = self.model(
                        batch["history"].to(self.device),
                        batch["static"].to(self.device),
                        batch["future_known"].to(self.device),
                    )
                    val_loss += criterion(pred, batch["target"].to(self.device)).item() * len(batch["target"])

            val_loss /= max(len(val_samples), 1)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self._test_samples = _build_samples(
            df,
            df["date"] == (split.test_start - pd.Timedelta(days=1)),
            cfg.history_length,
            cfg.horizon,
            sample_stride=1,
        )
        return self

    def predict_test(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        self.model.eval()
        rows = []
        loader = DataLoader(SeriesDataset(self._test_samples), batch_size=self.config.batch_size, shuffle=False)

        idx = 0
        with torch.no_grad():
            for batch in loader:
                pred = self.model(
                    batch["history"].to(self.device),
                    batch["static"].to(self.device),
                    batch["future_known"].to(self.device),
                ).cpu().numpy()
                batch_size = pred.shape[0]
                for b in range(batch_size):
                    sample = self._test_samples[idx + b]
                    for h in range(self.config.horizon):
                        rows.append(
                            {
                                "store_id": sample["store_id"],
                                "sku_id": sample["sku_id"],
                                "date": sample["target_dates"][h],
                                "origin_date": sample["origin_date"],
                                "horizon": h + 1,
                                "prediction": max(float(pred[b, h]), 0.0),
                                "units_sold": float(sample["target"][h]),
                                "channel": sample["channel"],
                                "category": sample["category"],
                                "promo_flag": sample["promo_flags"][h],
                                "purchase_cost": sample["purchase_cost"][h],
                                "margin_pct": sample["margin_pct"][h],
                            }
                        )
                idx += batch_size

        return pd.DataFrame(rows)
