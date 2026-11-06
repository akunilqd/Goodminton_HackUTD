"""M2: isolated upstream TrackNetV3 + InpaintNet inference and frame alignment.

The subprocess boundary avoids upstream generic `model`/`dataset` import collisions
and supports a separate Python environment for the research implementation.
"""

import csv
import math
import os
import subprocess
from pathlib import Path

from config import PROJECT_DIR, Settings, require_file
from schemas import Shuttle


def read_predictions(path: Path) -> dict[int, tuple[float, float] | None]:
    predictions = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"Frame", "Visibility", "X", "Y"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected Frame,Visibility,X,Y columns")
        for row in reader:
            index = int(row["Frame"])
            visibility = int(row["Visibility"])
            x, y = float(row["X"]), float(row["Y"])
            if index < 0 or index in predictions:
                raise ValueError(f"{path}: duplicate or negative frame {index}")
            if visibility not in (0, 1) or not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"{path}: invalid visibility/coordinates at frame {index}")
            predictions[index] = (x, y) if visibility else None
    if not predictions:
        raise ValueError(f"{path}: no shuttle predictions")
    if set(predictions) != set(range(len(predictions))):
        raise ValueError(f"{path}: frame indices must be contiguous and zero-based")
    return predictions


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
        if "--device" not in source or "--half" not in source:
            raise ValueError("tracknet_repo/predict.py must support --device and --half")
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

    def predict_video(self, video: Path, output_dir: Path) -> dict[int, Shuttle]:
        self.validate()
        video = require_file(video, "video")
        if video.suffix.lower() != ".mp4":
            raise ValueError("The upstream TrackNet adapter requires an .mp4 input")
        repo = self.settings.tracknet_repo.resolve()
        results = []
        for stage in ("raw", "inpainted"):
            stage_dir = (output_dir / stage).resolve()
            stage_dir.mkdir(parents=True, exist_ok=False)
            # Absolute forward-slash paths also satisfy upstream Windows filename parsing.
            command = [
                self.settings.tracknet_python,
                "-B",
                str(repo / "predict.py"),
                "--device",
                self.settings.device,
                "--video_file",
                video.as_posix(),
                "--tracknet_file",
                self.settings.tracknet_weights.resolve().as_posix(),
                "--save_dir",
                stage_dir.as_posix(),
                "--batch_size",
                str(self.settings.tracknet_batch_size),
                "--eval_mode",
                "nonoverlap",
                "--large_video",
            ]
            if self.settings.gpu_half:
                command.append("--half")
            if stage == "inpainted":
                command += [
                    "--inpaintnet_file",
                    self.settings.inpaintnet_weights.resolve().as_posix(),
                ]
            log_path = stage_dir / "inference.log"
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command,
                        cwd=repo,
                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=self.settings.tracknet_timeout_seconds,
                    )
            except (subprocess.SubprocessError, OSError) as exc:
                raise RuntimeError(f"TrackNet {stage} inference failed. See {log_path}") from exc
            results.append(read_predictions(stage_dir / f"{video.stem}_ball.csv"))
        return merge_predictions(*results)
