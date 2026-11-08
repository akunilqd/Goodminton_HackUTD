"""Command-line entry point and local preflight checks."""

import argparse
import importlib.util
import json
import logging
from dataclasses import asdict, replace
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
    if "4" in modules and not {"1", "2", "3"}.issubset(modules):
        errors.append("Video M4 requires --modules 1 2 3 4; use detect-hits to replay an existing run.")
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
    if "3" in modules:
        from module3_court.detector import ManualCourt

        if settings.court_calibration:
            try:
                ManualCourt(settings.court_calibration, settings)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"court_calibration: {exc}")
        else:
            packages.add("safetensors")
            if not settings.court_weights:
                packages.add("huggingface_hub")
            else:
                files.append((Path(settings.court_weights), "court_weights"))
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
    parser = argparse.ArgumentParser(description="Badminton analysis: Modules 1-4")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "check"):
        item = sub.add_parser(command)
        item.add_argument("--modules", nargs="+", choices=["1", "2", "3", "4"], default=list(settings.modules))
        if command == "run":
            item.add_argument("video", type=Path, nargs="?", default=settings.video)
            item.add_argument("--court-init-frame", type=int, default=settings.court_init_frame)
            item.add_argument("--output", type=Path, default=settings.output, help="New output directory")
    inspect = sub.add_parser("court-frame", help="Export numbered court keypoints for calibration")
    inspect.add_argument("video", type=Path, nargs="?", default=settings.video)
    inspect.add_argument("--frame", type=int, default=0)
    inspect.add_argument("--output", type=Path, required=True, help="New output directory")
    hits = sub.add_parser("detect-hits", help="Run CPU-only M4 on a complete saved M1/M2 run")
    hits.add_argument("run_dir", type=Path)
    hits.add_argument("--output", type=Path, required=True, help="New output directory")
    hits.add_argument("--rallies", type=Path, help="Optional JSON list of inclusive rally frame ranges")
    review = sub.add_parser("annotate-hits", help="Mark hit JSONL timestamps on a video without inference")
    review.add_argument("video", type=Path)
    review.add_argument("--hits", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True, help="New .mp4 output file")
    review.add_argument("--hold-seconds", type=float, default=0.16)
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

    from module3_court import CourtDetector, ManualCourt
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
            detector = (ManualCourt(settings.court_calibration, settings)
                        if settings.court_calibration else CourtDetector(settings))
            result = detector.process(frame)
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(
                args.output / "court.json",
                {
                    "frame_index": index,
                    "image_size": [reader.info.width, reader.info.height],
                    **asdict(result),
                },
            )
            from pipeline import render_frame
            from schemas import FrameResult
            render_frame(frame, FrameResult(index, index / reader.info.fps, court=result), [],
                         court_threshold=settings.court_keypoint_confidence)
            if not cv2.imwrite(str(args.output / "court.jpg"), frame):
                raise RuntimeError("Could not write court image")
            return
    raise ValueError(f"Video has no frame {args.frame}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        settings = Settings()
        args = build_parser(settings).parse_args(argv)
        if args.command == "annotate-hits":
            from pipeline import annotate_hits

            result = annotate_hits(args.video, args.hits, args.output, args.hold_seconds)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "detect-hits":
            from module4 import detect_saved_run

            summary = detect_saved_run(settings, args.run_dir, args.output, args.rallies)
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "court-frame":
            court_frame(settings, args)
            print(f"Court inspection saved to {args.output.resolve()}")
            return 0
        if args.command == "run":
            settings = replace(settings, court_init_frame=args.court_init_frame)
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
        from module4 import HitDetector

        hits = HitDetector(settings) if "4" in modules else None
        output = Pipeline(players, shuttle, court, hits).run(args.video, args.output)
        print(f"Analysis saved to {output}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
