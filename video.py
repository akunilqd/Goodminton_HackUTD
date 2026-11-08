"""Bounded-memory video decoding with explicit frame indices and CFR timestamps."""

import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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


def read_ahead(frames, pool, depth=16):
    """Yield `frames` in order while single-threaded `pool` decodes the next `depth`."""
    source = iter(frames)
    pending = deque(pool.submit(next, source, None) for _ in range(depth))
    while (item := pending.popleft().result()) is not None:
        pending.append(pool.submit(next, source, None))
        yield item


class BackgroundWriter:
    """Own an H.264 VideoWriter on one thread; work stays ordered and errors reach the caller."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], backlog=32):
        self.path, self.fps, self.size, self.backlog = path, fps, size, backlog
        self.pending = deque()

    def __enter__(self):
        self.pool = ThreadPoolExecutor(1)
        self.writer = self.pool.submit(
            cv2.VideoWriter, str(self.path), cv2.VideoWriter_fourcc(*"avc1"), self.fps, self.size,
        ).result()
        if not self.writer.isOpened():
            self.pool.shutdown()
            raise RuntimeError(f"Cannot open video writer: {self.path}")
        return self

    def submit(self, work, *args):
        """Queue `work(writer, *args)`, waiting when the encoder falls behind."""
        self.pending.append(self.pool.submit(work, self.writer, *args))
        while len(self.pending) > self.backlog:
            self.pending.popleft().result()

    def __exit__(self, error_type, *_):
        try:
            while error_type is None and self.pending:
                self.pending.popleft().result()
        finally:
            self.pool.submit(self.writer.release).result()
            self.pool.shutdown()
