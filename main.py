"""Command-line entry point and local preflight checks."""

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from config import DEFAULT_COURT_WEIGHTS, Settings, require_file


def configure_device(value: str) -> str:
    """Validate the requested backend instead of silently falling back to CPU."""
    import torch

    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable in this Python environment. Install CUDA-enabled "
                "PyTorch (see README.md), or explicitly set device='cpu'."
            )
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
        torch.cuda.set_device(index)
        # The models use fixed input sizes; reuse the fastest convolution kernels.
        torch.backends.cudnn.benchmark = True
        return f"cuda:{index}"
    if device.type != "cpu":
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:N'")
    return "cpu"


def preflight(settings: Settings, modules: set[str], *, use_saved_court=False):
    errors, warnings = [], []
    if "1" in modules and "3" not in modules:
        errors.append("Two-player court tracking requires Module 3; use --modules 1 3 (or 1 2 3).")
    packages = set()
    files = []
    devices = {}
    uses_court_model = "3" in modules and not (settings.court_calibration or use_saved_court)
    if "1" in modules or uses_court_model:
        try:
            device = configure_device(settings.device)
            for module in modules & {"1", "3"}:
                if module != "3" or uses_court_model:
                    devices[module] = device
        except (ImportError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
    if "1" in modules:
        packages.update(("ultralytics", "mmpose", "mmcv", "mmengine", "mmdet", "lap"))
        files += [
            (settings.rtmpose_config, "rtmpose_config"),
            (settings.rtmpose_weights, "rtmpose_weights"),
        ]
    if "2" in modules:
        from module2_shuttle import ShuttleTracker

        try:
            devices["2"] = ShuttleTracker(settings).validate()
        except ValueError as exc:
            errors.append(str(exc))
    if "3" in modules and not use_saved_court:
        from module3_court.detector import ManualCourt

        if settings.court_calibration:
            try:
                ManualCourt(settings.court_calibration, settings)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"court_calibration: {exc}")
        else:
            packages.add("safetensors")
            if settings.court_weights:
                weights = Path(settings.court_weights)
                if weights.suffix.lower() != ".safetensors":
                    errors.append("court_weights must be a CourtKeyNet .safetensors checkpoint")
                files.append((weights, "court_weights"))
            elif not DEFAULT_COURT_WEIGHTS.is_file():
                packages.add("huggingface_hub")
    for package in sorted(packages):
        if importlib.util.find_spec(package) is None:
            errors.append(f"Missing package: {package}; see README.md")
    for path, name in files:
        try:
            require_file(path, name)
        except ValueError as exc:
            errors.append(str(exc))
    return {"ok": not errors, "devices": devices, "errors": errors, "warnings": warnings}


def build_parser(settings):
    parser = argparse.ArgumentParser(description="Badminton analysis: Modules 1, 2 and 3")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "check"):
        item = sub.add_parser(command)
        item.add_argument("--modules", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
        if command == "run":
            item.add_argument("video", type=Path, nargs="?", default=settings.video)
            item.add_argument("--output", type=Path, default=settings.output,
                              help="New directory for the final annotated.mp4")
            item.add_argument("--court-detections", type=Path,
                              help="Reuse a fixed CourtKeyNet calibration from a prior run of this video")
    inspect = sub.add_parser("court-frame", help="Export numbered court keypoints for calibration")
    inspect.add_argument("video", type=Path, nargs="?", default=settings.video)
    inspect.add_argument("--frame", type=int, default=0)
    inspect.add_argument("--output", type=Path, required=True, help="New output directory")
    return parser


class SavedFixedCourt:
    """Supply a saved calibration to the tracking pipeline for the same clip."""

    def __init__(self, detections: Path, video: Path):
        import numpy as np

        from schemas import Court
        from video import VideoReader

        with detections.with_name("metadata.json").open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        with VideoReader(video) as reader:
            original = metadata["video"]
            if (Path(original["source"]).resolve() != reader.path
                    or (original["width"], original["height"]) != (reader.info.width, reader.info.height)):
                raise ValueError("--court-detections must come from this video at its current resolution")
        with detections.open(encoding="utf-8") as stream:
            record = json.loads(stream.readline())
        self.court = Court(**record["court"])
        matrix = np.asarray(self.court.homography, dtype=float)
        points = np.asarray(self.court.keypoints, dtype=float)
        if (metadata["status"] != "complete"
                or self.court.status not in {"fixed_courtkeynet", "manual_calibration"}
                or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or np.linalg.matrix_rank(matrix) != 3
                or points.ndim != 2 or points.shape[1] != 3 or len(points) < 4
                or not np.isfinite(points).all()):
            raise ValueError("--court-detections requires a completed fixed calibration run")

    def process(self, frame):
        return self.court


def court_frame(settings, args):
    import cv2

    from module3_court import CourtDetector
    from pipeline import write_json
    from video import VideoReader

    if args.frame < 0:
        raise ValueError("--frame must be non-negative")
    if args.output.exists():
        raise ValueError("Output directory already exists; select a new directory")
    with VideoReader(args.video) as reader:
        for index, _, frame in reader:
            if index != args.frame:
                continue
            result = CourtDetector(settings).process(frame)
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(
                args.output / "court.json",
                {
                    "frame_index": index,
                    "image_size": [reader.info.width, reader.info.height],
                    **asdict(result),
                },
            )
            for point_id, (x, y, score) in enumerate(result.keypoints):
                if score < settings.court_keypoint_confidence:
                    continue
                point = (round(x), round(y))
                cv2.circle(frame, point, 4, (0, 255, 255), -1)
                cv2.putText(
                    frame, str(point_id), point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
            if not cv2.imwrite(str(args.output / "court.jpg"), frame):
                raise RuntimeError("Could not write court image")
            return
    raise ValueError(f"Video has no frame {args.frame}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        settings = Settings()
        argv = sys.argv[1:] if argv is None else argv
        args = build_parser(settings).parse_args(argv or ["run"])
        if args.command == "court-frame":
            court_frame(settings, args)
            print(f"Court inspection saved to {args.output.resolve()}")
            return 0
        modules = set(args.modules)
        saved_court = None
        if args.command == "run" and args.court_detections is not None:
            if "3" not in modules:
                raise ValueError("--court-detections requires Module 3")
            if settings.court_calibration:
                raise ValueError("Choose either --court-detections or court_calibration")
            saved_court = SavedFixedCourt(args.court_detections, args.video)
        report = preflight(settings, modules, use_saved_court=saved_court is not None)
        if args.command == "check":
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
        require_file(args.video, "video")
        if args.output.exists():
            raise ValueError("Output directory already exists; select a new directory")
        for warning in report["warnings"]:
            logging.warning(warning)
        if report["errors"]:
            raise ValueError("\n".join(report["errors"]))
        logging.info("Inference devices: %s", report["devices"])
        from module1_players import PlayerTracker
        from module2_shuttle import ShuttleTracker
        from module3_court import CourtDetector, ManualCourt
        from pipeline import Pipeline

        players = PlayerTracker(settings) if "1" in modules else None
        shuttle = ShuttleTracker(settings) if "2" in modules else None
        court = saved_court
        if "3" in modules and court is None:
            court = (
                ManualCourt(settings.court_calibration, settings)
                if settings.court_calibration
                else CourtDetector(settings)
            )
        output = Pipeline(players, shuttle, court).run(args.video, args.output)
        print(f"Video saved to {output}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
