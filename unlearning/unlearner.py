"""
pipeline/unlearning/unlearner.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``Unlearner`` — the single public entry point for running a machine unlearning
experiment.

Responsibilities
----------------
* Accept a loaded model + two DataLoaders + a config.
* Deep-copy and LoRA-adapt the model (base model is never mutated).
* Run the training loop.
* Return the trained PEFT model.

Everything else is delegated:
  losses   → pipeline.unlearning.losses.unlearning_loss
  LoRA     → pipeline.unlearning.adapters.lora_adapter
  logging  → pipeline.unlearning.utils.logging

Extensibility
-------------
Override ``_train_step`` in a subclass to implement a different unlearning
algorithm (SCRUB, SISA, noise-label, …) without touching the epoch loop,
optimizer, or logging.
"""

from __future__ import annotations

import copy
import logging
from itertools import cycle
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline.unlearning.adapters.lora_adapter import LoraModelAdapter
from pipeline.unlearning.config import UnlearnerConfig
from pipeline.unlearning.losses.unlearning_loss import combined_loss
from pipeline.unlearning.utils.logging import UnlearningLogger

logger = logging.getLogger(__name__)


class Unlearner:
    """
    Gradient-ascent / retain-loss dual-objective machine unlearner.

    Parameters
    ----------
    model:
        Any ``nn.Module``.  Deep-copied internally — the original is
        never modified.
    forget_loader:
        DataLoader for samples the model should forget.
    retain_loader:
        DataLoader for samples the model should keep performing well on.
    config:
        ``UnlearnerConfig``.  Defaults to ``UnlearnerConfig()`` if not
        provided.

    Example
    -------
    >>> from pipeline.unlearning import Unlearner, UnlearnerConfig
    >>> cfg = UnlearnerConfig(epochs=8, lr=3.18e-4, rank=8, alpha=41,
    ...                       lambda_retain=1.36, device="cuda")
    >>> unlearner = Unlearner(model=model, forget_loader=f_loader,
    ...                       retain_loader=r_loader, config=cfg)
    >>> trained = unlearner.train()
    >>> trained.save_pretrained("./checkpoints/unlearned")
    """

    def __init__(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        config: Optional[UnlearnerConfig] = None,
    ) -> None:
        self.config = config or UnlearnerConfig()
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.device = torch.device(self.config.device)

        self._run_logger = UnlearningLogger(self.config)
        self._model = self._build_model(model)
        self._optimizer = self._build_optimizer()

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> nn.Module:
        """Run the full unlearning loop and return the trained model."""
        logger.info(
            "Starting unlearning | forget=%d  retain=%d",
            len(self.forget_loader.dataset),
            len(self.retain_loader.dataset),
        )
        forget_iter = cycle(self.forget_loader)
        global_step = 0

        for epoch in range(1, self.config.epochs + 1):
            avg_loss, avg_forget_prob = self._run_epoch(forget_iter, global_step)
            global_step += len(self.retain_loader)

            logger.info(
                "Epoch %d/%d | avg_forget_prob=%.6f (target<%.4f) | loss=%.4f",
                epoch, self.config.epochs,
                avg_forget_prob, self.config.forget_threshold, avg_loss,
            )
            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Avg Forget Prob: {avg_forget_prob:.6f} "
                f"(Target < {self.config.forget_threshold}) | "
                f"Loss: {avg_loss:.4f}"
            )

            if avg_forget_prob < self.config.early_stop_prob:
                logger.info(
                    "Early stopping: forget_prob %.6f < %.4f",
                    avg_forget_prob, self.config.early_stop_prob,
                )
                print("✅ Target unlearning threshold reached. Stopping early.")
                break

        self._run_logger.finish()
        return self._model

    # ── Overridable hook ──────────────────────────────────────────────────────

    def _train_step(
        self,
        f_images: torch.Tensor,
        f_labels: torch.Tensor,
        r_images: torch.Tensor,
        r_labels: torch.Tensor,
    ) -> tuple[float, float]:
        """
        One gradient update.  Override to swap in a different algorithm.

        Returns
        -------
        (loss_scalar, batch_forget_prob)
        """
        f_logits = self._model(f_images)
        r_logits = self._model(r_images)

        loss, batch_forget_prob = combined_loss(
            f_logits, f_labels,
            r_logits, r_labels,
            self.config.lambda_retain,
            self.config.forget_threshold,
        )

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item(), batch_forget_prob

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_epoch(
        self, forget_iter, global_step: int
    ) -> tuple[float, float]:
        total_loss = total_forget_prob = 0.0
        n = len(self.retain_loader)

        for r_images, r_labels in self.retain_loader:
            f_images, f_labels = next(forget_iter)

            f_images = f_images.to(self.device, non_blocking=True)
            f_labels = f_labels.to(self.device, non_blocking=True)
            r_images = r_images.to(self.device, non_blocking=True)
            r_labels = r_labels.to(self.device, non_blocking=True)

            step_loss, step_forget_prob = self._train_step(
                f_images, f_labels, r_images, r_labels
            )
            total_loss += step_loss
            total_forget_prob += step_forget_prob

            self._run_logger.log(
                {"step/loss": step_loss, "step/forget_prob": step_forget_prob},
                step=global_step,
            )
            global_step += 1

        avg_loss = total_loss / n
        avg_forget_prob = total_forget_prob / n
        self._run_logger.log(
            {"epoch/loss": avg_loss, "epoch/forget_prob": avg_forget_prob},
            step=global_step,
        )
        return avg_loss, avg_forget_prob

    def _build_model(self, raw_model: nn.Module) -> nn.Module:
        logger.info("Cloning model and injecting LoRA…")
        cloned = copy.deepcopy(raw_model)
        adapted = LoraModelAdapter(cloned, self.config).inject()
        return adapted.to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable, lr=self.config.lr, weight_decay=self.config.weight_decay
        )
