# Machine Unlearning Pipeline

A modular, architecture-agnostic machine unlearning framework built on PyTorch, timm, and PEFT.

This project enables efficient and reproducible machine unlearning using parameter-efficient fine-tuning (LoRA), with support for modern vision models including CNNs, Vision Transformers, and custom architectures.

The pipeline is designed to scale from small experiments to large-scale datasets and models, while remaining flexible and easy to extend.

---

# Features

* Architecture-agnostic — works with ResNet, VGG, MobileNet, ViT, and custom models
* Efficient unlearning via LoRA adapters (no full model retraining required)
* Modular design with clean separation of data, training, and evaluation
* Fast dataset handling with optimized loaders
* Compatible with timm, HuggingFace datasets, and standard PyTorch workflows
* Built-in evaluation utilities for measuring forgetting and retention
* Optional TensorBoard and Weights & Biases logging
* Designed for reproducible research and experimentation

---

# Installation

Clone the repository:

```bash
git clone https://github.com/sayantan-2/unlearning-pipeline.git
cd unlearning-pipeline
```

Install dependencies:

```bash
pip install torch torchvision timm peft datasets accelerate tensorboard
```

Optional (for experiment tracking):

```bash
pip install wandb
```

---

# Quick Start

Example using a timm model:

```python
import timm
import torch

from pipeline.unlearning import Unlearner, UnlearnerConfig

# Load model
model = timm.create_model("resnet50", pretrained=True)

# Define your own DataLoaders
forget_loader = ...
retain_loader = ...

# Configure unlearning
config = UnlearnerConfig(
    epochs=5,
    lr=2e-4,
    rank=8,
    alpha=32,
    lambda_retain=4.0,
    device="cuda",
    target_modules=["layer4.2.conv3", "fc"]
)

# Run unlearning
unlearner = Unlearner(
    model=model,
    forget_loader=forget_loader,
    retain_loader=retain_loader,
    config=config,
)

trained_model = unlearner.train()

# Save adapter
trained_model.save_pretrained("./checkpoints/unlearned")
```

---

# Design Philosophy

This framework is built around a few key principles:

**Architecture-agnostic**
Works with any PyTorch model without requiring architecture-specific code.

**Parameter-efficient**
Uses LoRA adapters to minimize memory usage and training time.

**Modular and extensible**
Components can be easily replaced or extended to implement new unlearning algorithms.

**Reproducible**
Explicit configuration and deterministic workflows ensure reliable experiments.

---

# Future Scope

This pipeline is designed to support a wide range of models and applications, including:

* Vision Transformers (ViT)
* Face recognition models
* Custom research architectures
* Large-scale datasets beyond ImageNet

---

# License

MIT License

---
