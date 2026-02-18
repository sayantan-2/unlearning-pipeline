"""
pipeline/unlearning/config.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All hyperparameters live here. Nothing else does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UnlearnerConfig:
    """
    Complete configuration for a single unlearning run.

    All fields have sensible defaults so you only need to specify
    what you actually want to change.

    Example
    -------
    >>> cfg = UnlearnerConfig(epochs=8, lr=3.18e-4, rank=8, alpha=41,
    ...                       lambda_retain=1.36, device="cuda")
    """

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.01

    # ── Loss ──────────────────────────────────────────────────────────────────
    lambda_retain: float = 1.0
    # Per-batch threshold: skip gradient ascent if forget prob already below this.
    forget_threshold: float = 0.001
    # Epoch-level early stop: halt training once epoch-avg forget prob < this.
    early_stop_prob: float = 0.005

    # ── LoRA ──────────────────────────────────────────────────────────────────
    rank: int = 16
    alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=list)

    # ── Runtime ───────────────────────────────────────────────────────────────
    device: str = "cuda"

    # ── Logging ───────────────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "machine-unlearning"
    wandb_run_name: Optional[str] = None

    use_tensorboard: bool = False
    tensorboard_log_dir: str = "./runs"
