# Machine Unlearning Pipeline

A clean, production-quality ImageNet unlearning framework built on timm + PEFT.

---

## Folder Structure

```
pipeline/
├── __init__.py
│
├── data/
│   ├── dataset.py          ← ImageNetParquetDataset (thin HF→PyTorch wrapper)
│   └── manager.py          ← ImageNetDataManager   (load-once, split-fast)
│
├── evaluation/
│   └── evaluator.py        ← ImageNetEvaluator     (run-once, query-any-metric)
│
├── unlearning/
│   ├── config.py           ← UnlearnerConfig       (all hyperparams, no logic)
│   ├── unlearner.py        ← Unlearner             (orchestrator)
│   ├── adapters/
│   │   └── lora_adapter.py ← LoraModelAdapter      (arch-agnostic LoRA injection)
│   ├── losses/
│   │   └── unlearning_loss.py  ← Pure loss functions
│   └── utils/
│       └── logging.py      ← TensorBoard + W&B façade
│
└── unlearning_pipeline.ipynb   ← Clean notebook (all experiments)
```

---

## Quick Start

```python
import timm, torch
from pipeline.data import ImageNetDataManager
from pipeline.evaluation import ImageNetEvaluator
from pipeline.unlearning import Unlearner, UnlearnerConfig

# 1. Model
model = timm.create_model("resnet50.a1h_in1k", pretrained=True)
transforms = timm.data.create_transform(
    **timm.data.resolve_model_data_config(model), is_training=False
)

# 2. Data  (load once)
dm = ImageNetDataManager("train/*.parquet", "val/*.parquet")
val_loader, f_loader, r_loader = dm.get_loaders(forget_class=9, transforms=transforms)

# 3. Baseline
ev = ImageNetEvaluator(model, val_loader, device="cuda", forget_class=9)
ev.run()
print(ev.summary())

# 4. Unlearn
config = UnlearnerConfig(epochs=8, lr=3.18e-4, rank=8, alpha=41,
                          lambda_retain=1.37, device="cuda")
trained = Unlearner(model=model, forget_loader=f_loader,
                    retain_loader=r_loader, config=config).train()
trained.save_pretrained("./checkpoints/unlearned")
```

---

## Key Design Decisions

### No `model_name` argument
The `Unlearner` accepts a loaded `nn.Module` and deep-copies it internally.
This avoids re-downloading weights on every trial and keeps model creation
outside the package.

### Architecture-agnostic LoRA targeting
`LoraModelAdapter` walks the live model graph at runtime and collects unique
`nn.Linear` / `nn.Conv2d` leaf names as LoRA targets.  No brittle
`if "resnet" elif "vit"` branches.  Set `UnlearnerConfig.target_modules`
explicitly for surgical control.

### Evaluator is run-once / query-many
`ImageNetEvaluator.run()` is called once; all metric methods (`top1_accuracy`,
`forget_entropy`, `summary()`, …) query the cached tensors.  Calling any
metric before `run()` raises a clear `RuntimeError`.

### `summary()` convenience method
Matches the pattern in the original notebook:
```python
print(json.dumps(ev.summary(), indent=4))
```

---

## Logging

```python
# TensorBoard
cfg = UnlearnerConfig(use_tensorboard=True, tensorboard_log_dir="./runs")

# W&B
cfg = UnlearnerConfig(use_wandb=True, wandb_project="unlearning-experiments",
                      wandb_run_name="resnet50-class9")
```

---

## Extending the Algorithm

Override `_train_step` to swap in a different unlearning objective:

```python
from pipeline.unlearning import Unlearner

class KLUnlearner(Unlearner):
    def _train_step(self, f_images, f_labels, r_images, r_labels):
        # Your KL-divergence / SCRUB / noise-label logic here
        ...
        return loss.item(), batch_forget_prob
```
