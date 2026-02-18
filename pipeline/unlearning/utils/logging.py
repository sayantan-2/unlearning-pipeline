"""
pipeline/unlearning/utils/logging.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thin façade over TensorBoard and W&B — the trainer only calls
``logger.log(metrics, step)`` and never imports either backend directly.
Both are optional and no-op gracefully if not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

std_logger = logging.getLogger(__name__)


class UnlearningLogger:
    def __init__(self, config: Any) -> None:  # config: UnlearnerConfig
        self._tb = None
        self._wandb = False

        if config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=config.tensorboard_log_dir)
                std_logger.info("TensorBoard → %s", config.tensorboard_log_dir)
            except ImportError:
                std_logger.warning("tensorboard not installed; skipping.")

        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=vars(config),
                )
                self._wandb = True
                std_logger.info("W&B enabled (project=%s)", config.wandb_project)
            except ImportError:
                std_logger.warning("wandb not installed; skipping.")

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self._tb:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, global_step=step)
        if self._wandb:
            import wandb
            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._tb:
            self._tb.close()
        if self._wandb:
            import wandb
            wandb.finish()
