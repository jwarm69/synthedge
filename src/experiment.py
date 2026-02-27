"""
Optional experiment tracking utilities (MLflow/W&B).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentTracker:
    def __init__(self, provider: str = "none", project: Optional[str] = None):
        self.provider = provider
        self.project = project
        self._run = None
        self._mlflow = None
        self._wandb = None

    def start(self, run_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        if self.provider == "mlflow":
            try:
                import mlflow  # type: ignore
                self._mlflow = mlflow
                mlflow.set_experiment(self.project or "crypto-kalshi-predictor")
                mlflow.start_run(run_name=run_name)
                if config:
                    mlflow.log_params(config)
            except Exception:
                self.provider = "none"
        elif self.provider == "wandb":
            try:
                import wandb  # type: ignore
                self._wandb = wandb
                self._run = wandb.init(
                    project=self.project or "crypto-kalshi-predictor",
                    name=run_name,
                    config=config
                )
            except Exception:
                self.provider = "none"

    def log_params(self, params: Dict[str, Any]) -> None:
        if self.provider == "mlflow" and self._mlflow:
            self._mlflow.log_params(params)
        elif self.provider == "wandb" and self._wandb:
            self._wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.provider == "mlflow" and self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)
        elif self.provider == "wandb" and self._wandb and self._run:
            self._run.log(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        if self.provider == "mlflow" and self._mlflow:
            self._mlflow.log_artifact(str(path))
        elif self.provider == "wandb" and self._wandb and self._run:
            self._run.save(str(path))

    def end(self) -> None:
        if self.provider == "mlflow" and self._mlflow:
            self._mlflow.end_run()
        elif self.provider == "wandb" and self._wandb and self._run:
            self._run.finish()


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return json.load(f)
