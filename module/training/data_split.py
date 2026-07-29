"""
Shared dataset construction and splitting for the four deployment combos.

Why this exists as its own module (SRP): every train_*.py script needs the
exact same answer to "what are the train/val/test images", and it must be
the *same* answer across all four combos — otherwise the four confusion
matrices and the leaderboard chart in the app (PRD §8.5) would each be
computed against a different test set and the comparison would be invalid.

Determinism guarantee: FullDataset enumerates files in a fixed filesystem
order, so `targets` (the label array) is identical every time it's built
from the same DATA_DIR, independent of `img_size`/`transform`. That means
train_test_split(..., random_state=42) and StratifiedKFold(..., random_state=42)
below select the *same indices* for every combo even though each backbone
decodes those images at a different resolution (e.g. tf_efficientnet_b4 at
380px vs 224px for the others, per module.models.RECOMMENDED_IMG_SIZES) —
same underlying images, different resize. This module exists specifically
to make that guarantee a documented contract rather than an accident four
separate scripts each get right (or don't) on their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset

from module.utils import FullDataset, get_base_transformations

RANDOM_SEED = 42  # must match train_swin.py's legacy-path seed — do not change per-combo


@dataclass
class SplitDataset:
    """Everything a training script needs to start looping over folds."""

    full_dataset: FullDataset
    class_names: list[str]
    class_weights_tensor: torch.FloatTensor
    targets: np.ndarray
    train_val_idx: np.ndarray
    test_idx: np.ndarray

    def kfold_splits(self, nfolds: int) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yields (fold_number_from_1, train_idx, val_idx) — indices into full_dataset."""
        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=RANDOM_SEED)
        tv_targets = self.targets[self.train_val_idx]
        for fold, (rel_tr, rel_val) in enumerate(skf.split(self.train_val_idx, tv_targets)):
            yield fold + 1, self.train_val_idx[rel_tr], self.train_val_idx[rel_val]


def load_and_split(data_dir: str, img_size: int, test_split: float, device: str) -> SplitDataset:
    """
    Build FullDataset at `img_size` resolution and compute the deterministic
    train_val / test partition. Call once per combo (img_size varies), the
    resulting *indices* will match across combos even though img_size doesn't.
    """
    transform = get_base_transformations(img_size)
    full_dataset = FullDataset(data_dir, transform)

    targets = np.array(full_dataset.targets)
    class_names = full_dataset.classes

    train_val_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=test_split,
        stratify=targets,
        random_state=RANDOM_SEED,
    )

    tv_targets = targets[train_val_idx]
    class_counts = np.bincount(tv_targets)
    total_samples = len(tv_targets)
    class_weights = total_samples / (len(class_names) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    return SplitDataset(
        full_dataset=full_dataset,
        class_names=class_names,
        class_weights_tensor=class_weights_tensor,
        targets=targets,
        train_val_idx=train_val_idx,
        test_idx=test_idx,
    )


def build_loader(
    full_dataset: FullDataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    """Thin wrapper kept consistent with train_swin.py's _build_legacy_loader."""
    return DataLoader(
        Subset(full_dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
