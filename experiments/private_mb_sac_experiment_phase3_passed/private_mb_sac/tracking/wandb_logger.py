"""Weights & Biases logging for private model-based SAC.

Design goals:
- no `wandb.login()` or `wandb.init()` at import time;
- one W&B run for the whole three-owner experiment;
- owner-specific metrics under `owner/{i}/...`;
- system aggregates under `system/...`;
- evaluation metrics use `round` as their explicit step;
- safe no-op behavior when tracking is disabled.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _to_python(value: Any) -> Any:
    """Convert JAX/NumPy scalars and small arrays to W&B-safe values."""
    if is_dataclass(value):
        return {key: _to_python(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_python(item) for item in value]

    try:
        array = np.asarray(value)
    except Exception:
        return value

    if array.ndim == 0:
        return array.item()
    return array.tolist()


class WandbLogger:
    """Single-run experiment logger with owner and system namespaces."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        entity: str | None,
        group: str | None,
        job_type: str,
        mode: str,
        name: str,
        tags: Sequence[str],
        config: Mapping[str, Any],
        output_dir: Path,
        save_code: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self._run = None

        if not self.enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B tracking is enabled but `wandb` is not installed. "
                "Install it with `pip install wandb`, or set "
                "`tracking.enabled: false`."
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)

        self._run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            job_type=job_type,
            mode=mode,
            name=name,
            tags=list(tags),
            config=_to_python(config),
            dir=str(output_dir),
            save_code=save_code,
        )
        self._define_metrics()

    @property
    def run(self):
        return self._run

    def _define_metrics(self) -> None:
        if not self.enabled:
            return
        import wandb

        wandb.define_metric("round")
        wandb.define_metric("system/*", step_metric="round")
        wandb.define_metric("owner/*", step_metric="round")
        wandb.define_metric("evaluation/*", step_metric="round")
        wandb.define_metric("diagnostics/*", step_metric="round")

    def log(
        self,
        metrics: Mapping[str, Any],
        *,
        round_index: int,
        commit: bool = True,
    ) -> None:
        if not self.enabled:
            return
        import wandb

        payload = {
            "round": int(round_index),
            **{str(key): _to_python(value) for key, value in metrics.items()},
        }
        wandb.log(payload, commit=commit)

    def log_owner(
        self,
        owner_id: int,
        metrics: Mapping[str, Any],
        *,
        round_index: int,
        commit: bool = False,
    ) -> None:
        prefix = f"owner/{owner_id}"
        self.log(
            {f"{prefix}/{key}": value for key, value in metrics.items()},
            round_index=round_index,
            commit=commit,
        )

    def log_system(
        self,
        metrics: Mapping[str, Any],
        *,
        round_index: int,
        commit: bool = True,
    ) -> None:
        self.log(
            {f"system/{key}": value for key, value in metrics.items()},
            round_index=round_index,
            commit=commit,
        )

    def log_evaluation(
        self,
        metrics: Mapping[str, Any],
        *,
        round_index: int,
        commit: bool = True,
    ) -> None:
        self.log(
            {f"evaluation/{key}": value for key, value in metrics.items()},
            round_index=round_index,
            commit=commit,
        )

    def log_histogram(
        self,
        key: str,
        values: Any,
        *,
        round_index: int,
        commit: bool = True,
    ) -> None:
        if not self.enabled:
            return
        import wandb

        array = np.asarray(values)
        wandb.log(
            {
                "round": int(round_index),
                key: wandb.Histogram(array),
            },
            commit=commit,
        )

    def log_artifact(
        self,
        *,
        path: Path,
        name: str,
        artifact_type: str,
        aliases: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        import wandb

        artifact = wandb.Artifact(
            name=name,
            type=artifact_type,
            metadata=_to_python(metadata or {}),
        )
        artifact.add_file(str(path))
        assert self._run is not None
        self._run.log_artifact(artifact, aliases=list(aliases))

    def finish(self) -> None:
        if not self.enabled:
            return
        import wandb

        wandb.finish()
        self._run = None
