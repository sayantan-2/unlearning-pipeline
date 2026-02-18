"""
pipeline/unlearning/losses/unlearning_loss.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pure loss functions — no trainer state, no I/O, easy to unit-test.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def forget_loss(
    logits: Tensor,
    labels: Tensor,
    batch_forget_prob: float,
    threshold: float,
) -> Tensor:
    """
    Gradient-ascent loss.  Zeroed if ``batch_forget_prob < threshold``
    (loss-capping — don't over-push already-unlearned batches).
    """
    if batch_forget_prob < threshold:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return -F.cross_entropy(logits, labels)


def retain_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Standard cross-entropy on the retain set."""
    return F.cross_entropy(logits, labels)


def combined_loss(
    f_logits: Tensor,
    f_labels: Tensor,
    r_logits: Tensor,
    r_labels: Tensor,
    lambda_retain: float,
    forget_threshold: float,
) -> tuple[Tensor, float]:
    """
    Combine forget and retain objectives.

    Returns
    -------
    (total_loss_tensor, batch_forget_prob_scalar)
    """
    probs = torch.softmax(f_logits, dim=1)
    batch_forget_prob = probs.gather(1, f_labels.view(-1, 1)).mean().item()

    f_loss = forget_loss(f_logits, f_labels, batch_forget_prob, forget_threshold)
    r_loss = retain_loss(r_logits, r_labels)

    return f_loss + lambda_retain * r_loss, batch_forget_prob
