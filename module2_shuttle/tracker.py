"""M2: isolated upstream TrackNetV3 + InpaintNet inference and frame alignment.

The subprocess boundary avoids upstream generic `model`/`dataset` import collisions
and supports a separate Python environment for the research implementation.
"""

import csv
import json
import math
import os
import subprocess
from pathlib import Path

from config import PROJECT_DIR, Settings, require_file
from schemas import Shuttle


def decode_predictions(columns, stage) -> dict[int, tuple[float, float] | None]:
    predictions = {}
    for index, visibility, x, y in zip(
        columns["Frame"], columns["Visibility"], columns["X"], columns["Y"], strict=True,
    ):
        if type(index) is not int or index < 0 or index in predictions:
            raise ValueError(f"{stage}: duplicate or invalid frame {index}")
        if visibility not in (0, 1) or not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"{stage}: invalid visibility/coordinates at frame {index}")
        predictions[index] = (float(x), float(y)) if visibility else None
    if not predictions:
        raise ValueError(f"{stage}: no shuttle predictions")
    if set(predictions) != set(range(len(predictions))):
        raise ValueError(f"{stage}: frame indices must be contiguous and zero-based")
    return predictions


def read_predictions(path: Path) -> dict[int, tuple[float, float] | None]:
    """Read a legacy TrackNet CSV exported by an older run."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = {key: [] for key in ("Frame", "Visibility", "X", "Y")}
        if not columns.keys() <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected Frame,Visibility,X,Y columns")
        for row in reader:
            for key in columns:
                convert = int if key in {"Frame", "Visibility"} else float
                columns[key].append(convert(row[key]))
    return decode_predictions(columns, path)


def merge_predictions(raw, repaired) -> dict[int, Shuttle]:
    if raw.keys() != repaired.keys():
        raise ValueError("TrackNet and InpaintNet frame indices do not match")
    samples = {}
    for index, pixel in repaired.items():
        changed = pixel != raw[index]
        source = "missing" if pixel is None else ("inpaintnet" if changed else "tracknet")
        samples[index] = Shuttle(
            pixel=pixel,
            raw_pixel=raw[index],
            visible=pixel is not None,
            rectified=changed,
            source=source,
        )
    return samples


class ShuttleTracker:
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate(self):
        script = require_file(
            self.settings.tracknet_repo / "predict.py", "tracknet_repo/predict.py"
        )
        require_file(self.settings.tracknet_weights, "tracknet_weights")
        require_file(self.settings.inpaintnet_weights, "inpaintnet_weights")
        source = script.read_text(encoding="utf-8")
        if "--inpaintnet_file" not in source:
            raise ValueError(
                "tracknet_repo must be qaz812345/TrackNetV3 (the InpaintNet implementation); "
                "alenzenx/TrackNetV3 is a different model"
            )
        # Check the actual subprocess interpreter, which may differ from main.py.
        try:
            result = subprocess.run(
                [self.settings.tracknet_python, "-B", "-c",
                 "import sys; from main import configure_device; "
                 "print(configure_device(sys.argv[1]))", self.settings.device],
                cwd=PROJECT_DIR, capture_output=True, text=True, check=True, timeout=60,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            raise ValueError(f"TrackNet Python/device check failed: {detail}") from exc
        return result.stdout.strip()

    def predict_video(self, video: Path) -> dict[int, Shuttle]:
        device = self.validate()
        video = require_file(video, "video")
        if video.suffix.lower() != ".mp4":
            raise ValueError("The upstream TrackNet adapter requires an .mp4 input")
        repo = self.settings.tracknet_repo.resolve()
        command = [
            self.settings.tracknet_python,
            "-B",
            str(Path(__file__).with_name("worker.py").resolve()),
            video.as_posix(),
            str(self.settings.tracknet_weights.resolve()),
            str(self.settings.inpaintnet_weights.resolve()),
            device,
            str(self.settings.tracknet_batch_size),
        ]
        if self.settings.gpu_half:
            command.append("--half")
        try:
            result = subprocess.run(
                command,
                cwd=repo,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=subprocess.PIPE,
                text=True,
                check=True,
                timeout=self.settings.tracknet_timeout_seconds,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(f"TrackNet/InpaintNet inference failed: {exc}") from exc
        try:
            raw, repaired = json.loads(result.stdout)
            return merge_predictions(
                decode_predictions(raw, "TrackNet"),
                decode_predictions(repaired, "InpaintNet"),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid shuttle predictions: {exc}") from exc
