"""Backward-compatible exports for tree boosting models."""

from src.models.baseline import (
    CatBoostConfig,
    CatBoostForecaster,
    LightGBMConfig,
    LightGBMForecaster,
    TreeBoostConfig,
    XGBoostConfig,
    XGBoostForecaster,
)

__all__ = [
    "TreeBoostConfig",
    "LightGBMConfig",
    "XGBoostConfig",
    "CatBoostConfig",
    "LightGBMForecaster",
    "XGBoostForecaster",
    "CatBoostForecaster",
]
