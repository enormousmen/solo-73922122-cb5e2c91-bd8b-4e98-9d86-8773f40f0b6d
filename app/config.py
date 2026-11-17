"""Training configuration handling.

A single :class:`TrainConfig` describes one training run.  The service loads a
default config from a JSON file (path from ``CONFIG_PATH`` env var, default
``config/default.json``); ``POST /train/start`` may override individual fields
(except ``dataset_path``, which is service-level).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1234
    epochs: int = 8
    batch_size: int = 32
    lr: float = 0.01
    hidden_dim: int = 16
    dropout_p: float = 0.2
    # Persist a checkpoint every N batches (and always at end of epoch).
    checkpoint_every: int = 1
    # Artificial per-batch delay in seconds (simulates slow training so that
    # pause / kill / query behaviour is observable in demos and tests).
    batch_sleep_seconds: float = 0.05
    # Artificial delay inside the checkpoint save, before the atomic rename
    # (simulates slow storage; used to prove saves never block queries and
    # that a kill mid-save can only leave a complete old or new checkpoint).
    checkpoint_sleep_seconds: float = 0.0
    # How long POST /train/pause waits for the trainer to stop and persist
    # a full checkpoint before reporting a timeout.
    pause_timeout_seconds: float = 30.0
    # Optional early stopping on the mean training loss of each epoch.
    # Disabled when early_stopping_patience is None.  When enabled, the first
    # epoch's mean loss becomes the initial best; a later epoch only resets
    # the no-improvement counter when (best - epoch_mean) > min_delta;
    # `patience` consecutive non-improving epochs stop the run.
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    # Service-level setting: CSV dataset used for every run.
    dataset_path: str = "data/classification.csv"

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0, 1)")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        if self.batch_sleep_seconds < 0:
            raise ValueError("batch_sleep_seconds must be >= 0")
        if self.checkpoint_sleep_seconds < 0:
            raise ValueError("checkpoint_sleep_seconds must be >= 0")
        if self.pause_timeout_seconds <= 0:
            raise ValueError("pause_timeout_seconds must be > 0")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        p = self.early_stopping_patience
        if p is not None:
            if isinstance(p, bool) or not isinstance(p, int):
                raise ValueError("early_stopping_patience must be an integer")
            if not 2 <= p <= 8:
                raise ValueError("early_stopping_patience must be in [2, 8]")
        md = self.early_stopping_min_delta
        if isinstance(md, bool) or not isinstance(md, (int, float)):
            raise ValueError("early_stopping_min_delta must be a number")
        if not math.isfinite(md) or md < 0:
            raise ValueError("early_stopping_min_delta must be finite and >= 0")
        if p is None and md != 0:
            raise ValueError(
                "early_stopping_min_delta requires early_stopping_patience "
                "to be set (early stopping is disabled otherwise)")

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def merged(self, overrides: dict | None) -> "TrainConfig":
        return TrainConfig.from_dict({**self.to_dict(), **(overrides or {})})


def load_default_config() -> TrainConfig:
    path = Path(os.environ.get("CONFIG_PATH", "config/default.json"))
    if path.exists():
        return TrainConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return TrainConfig()
