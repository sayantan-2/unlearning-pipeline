"""
pipeline/unlearning/adapters/lora_adapter.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Architecture-agnostic LoRA injection via PEFT.

Auto-detection strategy
-----------------------
Instead of maintaining a brittle ``if "resnet" elif "vit"`` lookup table,
we walk the live model graph and collect the *unique leaf-name suffixes* of
every ``nn.Linear`` and ``nn.Conv2d``.  PEFT matches these as suffixes, so
``"conv1"`` will match ``layer1.0.conv1``, ``layer2.0.conv1``, etc.

This works for any timm / torchvision / custom model with no changes.
Pass ``target_modules`` explicitly in ``UnlearnerConfig`` if you want
surgical control over which layers receive adapters.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch.nn as nn
from peft import LoraConfig, get_peft_model

from pipeline.unlearning.config import UnlearnerConfig

logger = logging.getLogger(__name__)


class LoraModelAdapter:
    """
    Injects LoRA adapters into a plain ``nn.Module`` and freezes all
    non-adapter parameters.

    Parameters
    ----------
    model:
        The base model to adapt (will be mutated in-place by PEFT).
    config:
        ``UnlearnerConfig`` — uses ``rank``, ``alpha``, ``lora_dropout``,
        and ``target_modules``.
    """

    def __init__(self, model: nn.Module, config: UnlearnerConfig) -> None:
        self._model = model
        self._config = config

    def inject(self) -> nn.Module:
        """Apply LoRA and return PEFT-wrapped model."""

        targets = self._config.target_modules

        if not targets:
            raise ValueError(
                "\nERROR: target_modules must be explicitly provided.\n\n"
                "Example:\n"
                "UnlearnerConfig(\n"
                "    target_modules=['layer4', 'fc']\n"
                ")\n\n"
                "Auto-detection is disabled by design.\n"
            )

        logger.info("Using user-provided LoRA targets: %s", targets)

        peft_cfg = LoraConfig(
            r=self._config.rank,
            lora_alpha=self._config.alpha,
            target_modules=targets,
            lora_dropout=self._config.lora_dropout,
            bias="none",
        )

        model = get_peft_model(self._model, peft_cfg)

        model.print_trainable_parameters()

        # Freeze everything except LoRA params
        model.eval()

        for name, param in model.named_parameters():
            param.requires_grad = "lora" in name

        return model
