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

Class weighting: uses effective-number-of-samples weighting (Cui et al.,
CVPR 2019 — see effective_number_weights() below) rather than raw
inverse-frequency, specifically because this dataset's class counts are
severely long-tailed (the majority class outnumbers the rarest by roughly
two orders of magnitude). Raw inverse-frequency (total / (num_classes *
count)) keeps growing without bound as a class gets rarer, which is exactly
what makes it fragile at this kind of imbalance ratio.
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

# Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (https://arxiv.org/abs/1901.05555).
# 0.999 is the value most commonly cited as a good default in that paper
# and in follow-up work for datasets with per-class counts in the
# hundreds-to-low-thousands range (this dataset's range).
EFFECTIVE_NUMBER_BETA = 0.999


def effective_number_weights(class_counts: np.ndarray, beta: float = EFFECTIVE_NUMBER_BETA) -> np.ndarray:
    """
    weight_i ∝ 1 / E_n_i, where E_n_i = (1 - beta^n_i) / (1 - beta) is the
    "effective number" of samples for class i.

    As n_i grows, E_n_i saturates toward 1/(1-beta) instead of growing
    linearly with n_i the way a raw count does — so, unlike raw
    inverse-frequency (whose weight shrinks as 1/n_i without bound), a class
    already past that saturation point gets essentially the same weight
    whether it has 3,000 samples or 30,000, while a genuinely rare class
    (n_i well below the saturation point) is still meaningfully up-weighted,
    just not by the full N/n_i ratio raw inverse-frequency would give it.

    Weights are normalized to sum to num_classes, the same scale convention
    (average weight ≈ 1) the previous raw-inverse-frequency version used.
    """
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(class_counts)
    return weights


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
    class_weights = effective_number_weights(class_counts)
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


def class_balanced_sampler(full_dataset: FullDataset, indices: np.ndarray) -> torch.utils.data.WeightedRandomSampler:
    """
    Per-sample weights (1 / count-of-that-sample's-class, restricted to
    `indices`) for a WeightedRandomSampler that draws roughly equal numbers
    of each class per epoch, regardless of the natural distribution.
    """
    targets = np.array(full_dataset.targets)[indices]
    counts = np.bincount(targets)
    per_sample_weight = 1.0 / counts[targets]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(per_sample_weight),
        num_samples=len(indices),
        replacement=True,
    )
