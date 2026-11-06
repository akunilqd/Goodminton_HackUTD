"""Bounded-memory video decoding with explicit frame indices and CFR timestamps."""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2

from config import require_file


@dataclass(frozen=True)
class VideoInfo:
    source: str
    width: int
    height: int
    fps: float
    reported_frame_count: int
    timebase: str = "frame_index / fps (constant frame rate)"


class VideoReader:
    def __init__(self, path: Path):
        self.path = require_file(path, "video")
        self.capture = None

    def __enter__(self):
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Cannot open video: {self.path}")
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            self.capture.release()
            raise ValueError("Video has invalid dimensions or frame rate")
        self.info = VideoInfo(
            str(self.path),
            width,
            height,
            fps,
            int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        return self

    def __iter__(self):
        if self.capture is None:
            raise RuntimeError("Use VideoReader as a context manager")
        index = 0
        while True:
            success, frame = self.capture.read()
            if not success:
                return
            yield index, index / self.info.fps, frame
            index += 1

    def __exit__(self, *_):
        self.capture.release()
        self.capture = None
