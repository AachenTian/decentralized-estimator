"""Minimal example of how the coordinator should log one round.

This file is documentation-by-code and is not the final training loop.
"""

from __future__ import annotations

from private_mb_sac.training.metrics import aggregate_owner_metrics
from private_mb_sac.tracking.wandb_logger import WandbLogger


def log_finished_round(
    logger: WandbLogger,
    *,
    round_index: int,
    owner_metrics: list[dict],
    system_real_steps_round: int,
    system_real_steps_total: int,
    round_seconds: float,
) -> None:
    for owner_id, metrics in enumerate(owner_metrics):
        logger.log_owner(
            owner_id,
            metrics,
            round_index=round_index,
            commit=False,
        )

    system_metrics = {
        "real_env_steps_round": system_real_steps_round,
        "real_env_steps_total": system_real_steps_total,
        "round_seconds": round_seconds,
        **aggregate_owner_metrics(owner_metrics),
    }
    logger.log_system(
        system_metrics,
        round_index=round_index,
        commit=True,
    )
