from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int = 0) -> int:
    """Seed Python ``random``, NumPy, and torch for reproducible runs.

    The online sidecar samples its replay buffer with Python's ``random`` and
    the synthetic data path uses NumPy, so seeding only ``torch`` left
    accept/reject outcomes non-reproducible.
    Returns the resolved integer seed for convenience/logging.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
