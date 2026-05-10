"""
============================================================================
Med-Image CompareNet — Dataset Loaders & Preprocessing
============================================================================
Handles loading and preprocessing for both medical imaging modalities:

  1. Chest X-Ray (Pneumonia vs Normal)
     - Kaggle: paultimothymooney/chest-xray-pneumonia
     - Structure: data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg

  2. Breast Histopathology (IDC — Invasive Ductal Carcinoma)
     - Kaggle: paultimothymooney/breast-histopathology-images
     - Structure: data/breast_histopathology/{patient_id}/{0,1}/*.png
       • 0 = benign, 1 = malignant (IDC positive)

Key Design Decisions:
  • Both datasets are resized to 224×224 for compatibility with pretrained models.
  • Identical train/val/test splits are used for CNN and ViT (fair comparison).
  • Data augmentation is applied only during training.
  • Class imbalance is handled via weighted sampling.
============================================================================
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

logger = logging.getLogger("MedImageCompareNet")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TRANSFORMS — Augmentation Pipelines                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_transforms(
    image_size: int = 224,
    is_training: bool = True,
    augmentation_cfg: Optional[Dict] = None,
) -> transforms.Compose:
    """
    Build a torchvision transform pipeline.

    Training: augmentation + normalization
    Validation/Test: resize + center-crop + normalization

    ImageNet normalization is used because our pretrained models (ResNet,
    DenseNet, ViT) were all trained on ImageNet.
    """
    # ImageNet statistics — these are the standard values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_training and augmentation_cfg:
        aug = augmentation_cfg
        transform_list = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5 if aug.get("horizontal_flip", True) else 0),
            transforms.RandomVerticalFlip(p=0.5 if aug.get("vertical_flip", False) else 0),
            transforms.RandomRotation(degrees=aug.get("rotation_range", 15)),
        ]

        cj = aug.get("color_jitter", {})
        if cj:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj.get("brightness", 0.2),
                    contrast=cj.get("contrast", 0.2),
                    saturation=cj.get("saturation", 0.1),
                    hue=cj.get("hue", 0.05),
                )
            )

        erasing_prob = aug.get("random_erasing_prob", 0.1)
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=erasing_prob),
        ])
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

    return transforms.Compose(transform_list)


def get_inverse_normalize() -> transforms.Normalize:
    """Inverse of ImageNet normalization — for visualizing tensor images."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CHEST X-RAY DATASET                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ChestXRayDataset(Dataset):
    """
    Chest X-Ray dataset for Pneumonia classification.

    Expected directory layout:
        root_dir/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── val/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── test/
            ├── NORMAL/
            └── PNEUMONIA/

    Each image is a grayscale or RGB JPEG.  Grayscale images are converted
    to 3-channel RGB to match pretrained model expectations.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = ["NORMAL", "PNEUMONIA"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.image_paths: List[str] = []
        self.labels: List[int] = []

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.image_paths.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])

        logger.info(
            f"ChestXRay [{split}]: {len(self)} images "
            f"({sum(1 for l in self.labels if l == 0)} normal, "
            f"{sum(1 for l in self.labels if l == 1)} pneumonia)"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss balancing."""
        counts = np.bincount(self.labels)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  BREAST HISTOPATHOLOGY DATASET                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class BreastHistopathologyDataset(Dataset):
    """
    IDC Breast Cancer Histopathology dataset.

    Expected layout:
        root_dir/
        ├── <patient_id>/
        │   ├── 0/    (benign patches)
        │   │   └── *.png
        │   └── 1/    (malignant / IDC-positive patches)
        │       └── *.png
        └── ...

    Each patch is a small 50×50 RGB image extracted from a whole-slide
    histopathology scan.  We resize to 224×224 for model input.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        max_per_class: int = 20000,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ["benign", "malignant"]
        self.class_to_idx = {"benign": 0, "malignant": 1}

        # Collect all image paths
        all_paths_0: List[str] = []  # benign
        all_paths_1: List[str] = []  # malignant

        for patient_dir in self.root_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            benign_dir = patient_dir / "0"
            malignant_dir = patient_dir / "1"

            if benign_dir.exists():
                all_paths_0.extend([str(p) for p in benign_dir.glob("*.png")])
            if malignant_dir.exists():
                all_paths_1.extend([str(p) for p in malignant_dir.glob("*.png")])

        # Cap per class to manage dataset size
        rng = np.random.RandomState(seed)
        if len(all_paths_0) > max_per_class:
            all_paths_0 = list(rng.choice(all_paths_0, max_per_class, replace=False))
        if len(all_paths_1) > max_per_class:
            all_paths_1 = list(rng.choice(all_paths_1, max_per_class, replace=False))

        all_paths = all_paths_0 + all_paths_1
        all_labels = [0] * len(all_paths_0) + [1] * len(all_paths_1)

        # Stratified train/val/test split
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_paths, all_labels, test_size=0.3, stratify=all_labels, random_state=seed
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
        )

        split_map = {
            "train": (train_paths, train_labels),
            "val": (val_paths, val_labels),
            "test": (test_paths, test_labels),
        }

        self.image_paths, self.labels = split_map[split]

        logger.info(
            f"BreastHistopathology [{split}]: {len(self)} patches "
            f"({sum(1 for l in self.labels if l == 0)} benign, "
            f"{sum(1 for l in self.labels if l == 1)} malignant)"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss balancing."""
        counts = np.bincount(self.labels)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADER FACTORY                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def create_dataloaders(
    config: Dict,
    dataset_name: str = "xray",
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for the specified dataset.

    Parameters
    ----------
    config : dict
        Master configuration dictionary.
    dataset_name : str
        "xray" or "pathology".

    Returns
    -------
    dict
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    data_cfg = config["data"][dataset_name if dataset_name == "xray" else "pathology"]
    cnn_cfg = config["cnn"]
    image_size = data_cfg["image_size"]
    batch_size = cnn_cfg["training"]["batch_size"]
    num_workers = cnn_cfg["training"]["num_workers"]
    seed = config["project"]["seed"]

    # Build transforms
    train_transform = get_transforms(image_size, is_training=True, augmentation_cfg=cnn_cfg["augmentation"])
    eval_transform = get_transforms(image_size, is_training=False)

    if dataset_name == "xray":
        train_ds = ChestXRayDataset(data_cfg["root_dir"], "train", train_transform)
        val_ds = ChestXRayDataset(data_cfg["root_dir"], "val", eval_transform)
        test_ds = ChestXRayDataset(data_cfg["root_dir"], "test", eval_transform)
    else:
        max_patches = data_cfg.get("max_patches_per_class", 20000)
        train_ds = BreastHistopathologyDataset(data_cfg["root_dir"], "train", train_transform, max_patches, seed)
        val_ds = BreastHistopathologyDataset(data_cfg["root_dir"], "val", eval_transform, max_patches, seed)
        test_ds = BreastHistopathologyDataset(data_cfg["root_dir"], "test", eval_transform, max_patches, seed)

    # Weighted sampler for imbalanced training data
    train_weights = train_ds.get_class_weights()
    sample_weights = [train_weights[l].item() for l in train_ds.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }

    return loaders
