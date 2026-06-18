"""Reproducibility helpers for experiments and audit scripts."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch when it is available.

    The helper is intentionally lightweight: importing it does not require
    torch. Entry-point scripts can call this once near startup.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    """Seed a PyTorch DataLoader worker from torch.initial_seed()."""

    try:
        import torch
    except Exception:
        return

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int = 42) -> Optional[object]:
    """Return a seeded torch.Generator when torch is installed."""

    try:
        import torch
    except Exception:
        return None

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
