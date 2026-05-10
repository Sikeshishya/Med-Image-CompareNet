"""
============================================================================
Med-Image CompareNet — Shared Utilities
============================================================================
Central utility functions used across all modules:
  • Configuration loading (YAML → Python dict)
  • Reproducibility (seed everything)
  • Device selection (CUDA / CPU)
  • Logging helpers
  • Image I/O helpers
============================================================================
"""

import os
import random
import logging
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load the master YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to config.yaml (relative or absolute).

    Returns
    -------
    dict
        Nested dictionary of all configuration values.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Auto-create output directories
    for key, path in config.get("paths", {}).items():
        os.makedirs(path, exist_ok=True)

    return config


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility across:
      • Python's `random` module
      • NumPy
      • PyTorch (CPU + CUDA)

    Parameters
    ----------
    seed : int
        The seed value.  Default = 42 (the answer to everything).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms (slight performance cost but 100% reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set to {seed} for full reproducibility.")


# ---------------------------------------------------------------------------
# Device Selection
# ---------------------------------------------------------------------------

def get_device(config: Dict[str, Any]) -> torch.device:
    """
    Select compute device based on config and hardware availability.

    Priority: CUDA GPU → CPU

    Parameters
    ----------
    config : dict
        The loaded config dictionary.

    Returns
    -------
    torch.device
        The selected device.
    """
    device_cfg = config.get("device", {})
    use_cuda = device_cfg.get("use_cuda", True)
    gpu_id = device_cfg.get("gpu_id", 0)

    if use_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        gpu_name = torch.cuda.get_device_name(gpu_id)
        logging.info(f"Using GPU: {gpu_name} (cuda:{gpu_id})")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU (CUDA not available or disabled in config).")

    return device


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Configure project-wide logging to both console and file.

    Parameters
    ----------
    log_dir : str
        Directory to save log files.
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"medimage_{timestamp}.log")

    # Create formatter
    fmt = "[%(asctime)s] %(levelname)s — %(name)s — %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Root logger
    logger = logging.getLogger("MedImageCompareNet")
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized.  Log file: {log_file}")
    return logger


# ---------------------------------------------------------------------------
# Image I/O Helpers
# ---------------------------------------------------------------------------

def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert grayscale image to 3-channel RGB (required by pretrained models)."""
    if len(image.shape) == 2:
        return np.stack([image] * 3, axis=-1)
    if image.shape[2] == 1:
        return np.concatenate([image] * 3, axis=-1)
    return image


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize any float/int image to uint8 [0, 255]."""
    img = image.astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# Checkpoint Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
) -> None:
    """
    Save a training checkpoint.

    Saved keys: model_state_dict, optimizer_state_dict, epoch, metrics.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )
    logging.info(f"Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Returns
    -------
    dict
        Dictionary with keys: epoch, metrics.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logging.info(f"Checkpoint loaded ← {path}  (epoch {checkpoint['epoch']})")
    return {"epoch": checkpoint["epoch"], "metrics": checkpoint["metrics"]}


# ---------------------------------------------------------------------------
# Metric Formatting
# ---------------------------------------------------------------------------

def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Pretty-print a metrics dictionary."""
    parts = [f"{prefix}{k}: {v:.4f}" for k, v in metrics.items()]
    return " | ".join(parts)
