"""Save and load production model artifacts with joblib."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.data.features import FeatureEncoder


@dataclass
class ProductionArtifacts:
    """Bundle everything needed for batch inference."""

    model: Any
    model_name: str
    encoder: FeatureEncoder
    feature_cols: list[str]
    model_feature_cols: list[str]
    horizon: int
    target: str
    group_cols: list[str]
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    train_data_end: str | None = None
    metadata: dict = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        bundle_path = directory / "production_bundle.joblib"
        meta_path = directory / "production_metadata.json"

        joblib.dump(self, bundle_path)

        meta = {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "target": self.target,
            "group_cols": self.group_cols,
            "feature_cols": self.feature_cols,
            "model_feature_cols": self.model_feature_cols,
            "trained_at": self.trained_at,
            "train_data_end": self.train_data_end,
            "bundle_file": bundle_path.name,
            **self.metadata,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return bundle_path

    @classmethod
    def load(cls, directory: str | Path) -> ProductionArtifacts:
        directory = Path(directory)
        bundle_path = directory / "production_bundle.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Production bundle not found: {bundle_path}")
        artifacts: ProductionArtifacts = joblib.load(bundle_path)
        return artifacts

    @classmethod
    def load_metadata(cls, directory: str | Path) -> dict:
        meta_path = Path(directory) / "production_metadata.json"
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return json.load(f)
