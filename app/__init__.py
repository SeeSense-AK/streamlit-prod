"""
SeeSense Safety Analytics Platform
Production-ready cycling safety dashboard
"""

__version__ = "1.0.0"
__author__ = "SeeSense"
__description__ = "Production cycling safety analytics dashboard"

# Package-level imports for easier access
from .utils.config import config
from .core.data_processor import data_processor

__all__ = [
    "config",
    "data_processor"
]
