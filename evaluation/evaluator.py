"""
pipeline/evaluation/evaluator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``ImageNetEvaluator`` — run inference once, query any metric lazily.

Changes from original
---------------------
* ``forget_class`` moves to ``__init__`` (not a magic constant).
* ``run()`` is idempotent — calling it twice is safe (re-runs inference).
* All metric methods raise ``RuntimeError`` with a helpful message if
  called before ``run()``, instead of silently returning wrong values.
* ``summary()`` returns a single dict so callers don't have to assemble
  results manually (matches the notebook pattern exactly).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm


class ImageNetEvaluator:
    """
    Runs a full forward pass over a DataLoader and caches predictions,
    probabilities, and logits for subsequent metric queries.

    Parameters
    ----------
    model:
        Any ``nn.Module`` in eval or train mode (we call ``.eval()``
        internally during ``run()``).
    loader:
        DataLoader yielding ``(images, labels)`` batches.
    device:
        Target device string, e.g. ``"cuda"`` or ``"cpu"``.
    forget_class:
        The class index treated as the *forget* target for
        forget-specific metrics.

    Example
    -------
    >>> ev = ImageNetEvaluator(model, val_loader, forget_class=9)
    >>> ev.run()
    >>> print(ev.summary())
    """

    def __init__(
        self,
        model,
        loader,
        device: str = "cuda",
        forget_class: int = 9,
    ) -> None:
        self.model = model
        self.loader = loader
        self.device = device
        self.forget_class = forget_class

        self._ran = False
        self.y_true: torch.Tensor
        self.y_pred: torch.Tensor
        self.logits: torch.Tensor
        self.probs: torch.Tensor

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def run(self) -> "ImageNetEvaluator":
        """Run inference and cache results.  Returns ``self`` for chaining."""
        self.model.eval()

        _y_true, _y_pred, _logits, _probs = [], [], [], []

        for images, labels in tqdm(self.loader, desc="Evaluating", leave=False):
            images = images.to(self.device, non_blocking=True)
            logits = self.model(images)
            probs = F.softmax(logits, dim=1)

            _y_true.append(labels)
            _y_pred.append(logits.argmax(dim=1).cpu())
            _logits.append(logits.cpu())
            _probs.append(probs.cpu())

        self.y_true = torch.cat(_y_true)
        self.y_pred = torch.cat(_y_pred)
        self.logits = torch.cat(_logits)
        self.probs = torch.cat(_probs)
        self._ran = True
        return self

    # ── Metrics ───────────────────────────────────────────────────────────────

    def top1_accuracy(self) -> float:
        self._check_ran()
        return (self.y_pred == self.y_true).float().mean().item()

    def top5_accuracy(self) -> float:
        self._check_ran()
        _, top5 = self.logits.topk(5, dim=1)
        correct = top5.eq(self.y_true.view(-1, 1).expand_as(top5))
        return correct.any(dim=1).float().mean().item()

    def retain_forget_accuracy(self) -> Dict[str, float]:
        self._check_ran()
        forget_mask = self.y_true == self.forget_class
        retain_mask = ~forget_mask

        forget_acc = (
            (self.y_pred[forget_mask] == self.forget_class).float().mean().item()
            if forget_mask.any() else 0.0
        )
        retain_acc = (
            (self.y_pred[retain_mask] == self.y_true[retain_mask]).float().mean().item()
            if retain_mask.any() else 0.0
        )
        return {"retain_acc": retain_acc, "forget_acc": forget_acc}

    def forget_confidence(self) -> float:
        """Mean softmax probability assigned to the forget class on forget samples."""
        self._check_ran()
        mask = self.y_true == self.forget_class
        if not mask.any():
            return 0.0
        return self.probs[mask, self.forget_class].mean().item()

    def forget_entropy(self) -> float:
        """Mean prediction entropy over forget samples."""
        self._check_ran()
        mask = self.y_true == self.forget_class
        if not mask.any():
            return 0.0
        p = self.probs[mask]
        return -(p * (p + 1e-9).log()).sum(dim=1).mean().item()

    def logit_margin(self) -> float:
        """
        Mean (logit_forget − max_other_logit) over forget samples.
        Negative → forget class is NOT the top prediction.
        """
        self._check_ran()
        mask = self.y_true == self.forget_class
        if not mask.any():
            return 0.0
        target_logits = self.logits[mask]
        forget_vals = target_logits[:, self.forget_class]
        masked = target_logits.clone()
        masked[:, self.forget_class] = -float("inf")
        return (forget_vals - masked.max(dim=1).values).mean().item()

    def per_class_accuracy(self) -> Dict[int, float]:
        """Vectorised per-class accuracy using ``torch.bincount``."""
        self._check_ran()
        num_classes = self.logits.size(1)
        correct_mask = self.y_pred == self.y_true
        total = torch.bincount(self.y_true, minlength=num_classes).float()
        correct = torch.bincount(
            self.y_true[correct_mask], minlength=num_classes
        ).float()
        safe_total = total.clamp(min=1.0)
        accuracies = correct / safe_total
        return {
            c: acc.item()
            for c, acc in enumerate(accuracies)
            if total[c] > 0
        }

    def summary(self) -> Dict[str, object]:
        """
        Convenience method that returns all scalar metrics in a single dict.
        Matches the pattern used in the notebook.

        Returns
        -------
        Dict with keys: top1, top5, retain_acc, forget_acc,
        forget_conf, forget_entropy, logit_margin.
        """
        self._check_ran()
        rf = self.retain_forget_accuracy()
        return {
            "top1": self.top1_accuracy(),
            "top5": self.top5_accuracy(),
            "retain_acc": rf["retain_acc"],
            "forget_acc": rf["forget_acc"],
            "forget_conf": self.forget_confidence(),
            "forget_entropy": self.forget_entropy(),
            "logit_margin": self.logit_margin(),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _check_ran(self) -> None:
        if not self._ran:
            raise RuntimeError("Call evaluator.run() before accessing metrics.")
