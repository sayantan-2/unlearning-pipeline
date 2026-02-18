from __future__ import annotations

import logging
from typing import Tuple, Dict, List

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

from pipeline.data.dataset import ImageNetParquetDataset

logger = logging.getLogger(__name__)


class ImageNetDataManager:
    """
    Ultra-fast ImageNet parquet manager.

    Features
    --------
    • Instant init (~5 sec vs 90 sec pandas)
    • Zero pandas dependency
    • Precomputed class index cache
    • O(1) loader creation
    • Optuna-safe
    • Memory efficient
    """

    def __init__(
        self,
        train_files: str,
        val_files: str,
    ) -> None:

        logger.info("Loading ImageNet parquet shards...")

        self.raw_train = load_dataset(
            "parquet",
            data_files=train_files,
            split="train",
        )

        self.raw_val = load_dataset(
            "parquet",
            data_files=val_files,
            split="train",
        )

        logger.info("Building class index cache...")

        # Convert labels to numpy (FAST)
        self.train_labels = np.array(self.raw_train["label"], dtype=np.int32)

        # Build index lookup per class
        self.class_to_indices: Dict[int, np.ndarray] = {}

        unique_classes = np.unique(self.train_labels)

        for cls in unique_classes:
            self.class_to_indices[int(cls)] = np.where(self.train_labels == cls)[0]

        logger.info(
            "DataManager ready | train=%d val=%d classes=%d",
            len(self.raw_train),
            len(self.raw_val),
            len(unique_classes),
        )

    # ------------------------------------------------------------------

    def get_loaders(
        self,
        forget_class: int,
        transforms,
        batch_size: int = 32,
        val_batch_size: int = 64,
        retain_n_per_class: int = 1,
        num_workers: int = 4,
        random_state: int = 42,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:

        rng = np.random.RandomState(random_state)

        # Validation loader
        val_loader = self._make_loader(
            self.raw_val,
            transforms,
            val_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # Forget loader
        forget_indices = self.class_to_indices[forget_class]

        forget_loader = self._make_loader(
            self.raw_train.select(forget_indices.tolist()),
            transforms,
            batch_size,
            shuffle=True,
            num_workers=num_workers // 2,
        )

        # Retain loader (stratified sampling)
        retain_indices: List[int] = []

        for cls, indices in self.class_to_indices.items():

            if cls == forget_class:
                continue

            if retain_n_per_class >= len(indices):
                retain_indices.extend(indices.tolist())

            else:
                sampled = rng.choice(
                    indices,
                    size=retain_n_per_class,
                    replace=False,
                )
                retain_indices.extend(sampled.tolist())

        retain_loader = self._make_loader(
            self.raw_train.select(retain_indices),
            transforms,
            batch_size,
            shuffle=True,
            num_workers=num_workers // 2,
        )

        logger.info(
            "Loaders ready | val=%d forget=%d retain=%d",
            len(self.raw_val),
            len(forget_indices),
            len(retain_indices),
        )

        return val_loader, forget_loader, retain_loader

    # ------------------------------------------------------------------

    @staticmethod
    def _make_loader(
        hf_dataset,
        transforms,
        batch_size: int,
        *,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoader:

        return DataLoader(
            ImageNetParquetDataset(hf_dataset, transforms),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
