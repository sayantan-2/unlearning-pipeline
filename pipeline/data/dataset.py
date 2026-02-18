"""
pipeline/data/dataset.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Low-level PyTorch Dataset wrapping a HuggingFace parquet dataset.

Kept intentionally thin — no transforms, no augmentation policy live here.
All that belongs in the caller (DataManager / transforms config).
"""

from __future__ import annotations

from torch.utils.data import Dataset


class ImageNetParquetDataset(Dataset):
    """
    Wraps a HuggingFace ``datasets.Dataset`` (loaded from parquet) so it
    can be used with a standard PyTorch ``DataLoader``.

    Parameters
    ----------
    hf_dataset:
        A ``datasets.Dataset`` instance with ``"image"`` and ``"label"`` columns.
        The ``"image"`` column must contain PIL Images or objects with a
        ``.convert()`` method (e.g. HuggingFace Image feature).
    transform:
        A callable applied to each PIL image (e.g. ``timm`` transform pipeline).
    """

    def __init__(self, hf_dataset, transform) -> None:
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        return self.transform(image), label
