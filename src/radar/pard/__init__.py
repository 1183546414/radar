"""PARD: Physics-Aware Robust Decision for radar anti-jamming.

This package is intentionally independent from the legacy VAE, retrieval, Q,
ensemble and ICDT implementations.  It consumes the raw CSV schema directly.
"""

from .config import PARDConfig
from .model import PARDModel

__all__ = ["PARDConfig", "PARDModel"]
