"""Offline hit detection from aligned player and shuttle observations."""

from .detector import HitDetector, detect_saved_run, write_hits

__all__ = ["HitDetector", "detect_saved_run", "write_hits"]
