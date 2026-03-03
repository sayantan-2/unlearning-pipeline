from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# Gradient Ascent (legacy)
# ============================================================


def gradient_ascent_loss(
    f_logits: Tensor,
    f_labels: Tensor,
    r_logits: Tensor,
    r_labels: Tensor,
    lambda_retain: float,
):
    forget_loss = -F.cross_entropy(f_logits, f_labels)
    retain_loss = F.cross_entropy(r_logits, r_labels)

    total = forget_loss + lambda_retain * retain_loss

    probs = torch.softmax(f_logits, dim=1)
    forget_prob = probs.gather(1, f_labels.view(-1, 1)).mean().item()

    return total, forget_prob


# ============================================================
# Hybrid Loss (NEW DEFAULT)
# ============================================================


def hinge_forget_loss(
    logits: Tensor,
    labels: Tensor,
    margin: float,
):
    true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

    logits_clone = logits.clone()
    logits_clone[range(len(labels)), labels] = -1e9
    max_other = logits_clone.max(dim=1).values

    loss = torch.relu(margin + true_logits - max_other)
    return loss.mean()


def masked_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    forget_class: int,
):
    student_log_probs = F.log_softmax(student_logits, dim=1)
    teacher_probs = F.softmax(teacher_logits, dim=1)

    mask = torch.ones_like(student_log_probs)
    mask[:, forget_class] = 0

    student_log_probs = student_log_probs * mask
    teacher_probs = teacher_probs * mask

    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def hybrid_loss(
    f_logits: Tensor,
    f_labels: Tensor,
    r_logits: Tensor,
    r_labels: Tensor,
    teacher_r_logits: Tensor,
    forget_class: int,
    lambda_hinge: float,
    lambda_kl: float,
    hinge_margin: float,
):
    hinge = hinge_forget_loss(f_logits, f_labels, hinge_margin)

    kl = masked_kl_loss(
        r_logits,
        teacher_r_logits,
        forget_class,
    )

    total = lambda_hinge * hinge + lambda_kl * kl

    probs = torch.softmax(f_logits, dim=1)
    forget_prob = probs.gather(1, f_labels.view(-1, 1)).mean().item()

    return total, forget_prob
