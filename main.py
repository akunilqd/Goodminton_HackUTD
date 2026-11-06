"""Command-line entry point and local preflight checks."""

import argparse
import importlib.util
import json
import logging
from dataclasses import asdict
from pathlib import Path

from config import Settings, require_file


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


def preflight(settings: Settings, modules: set[str]):
    errors, warnings = [], []
    if "1" in modules and "3" not in modules:
        errors.append("Two-player court tracking requires Module 3; use --modules 1 3 (or 1 2 3).")
    packages = set()
    files = []
    devices = {}
    if "1" in modules or ("3" in modules and not settings.court_calibration):
        try:
            device = configure_device(settings.device)
            for module in modules & {"1", "3"}:
                if module != "3" or not settings.court_calibration:
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
        from module3_court.detector import ManualCourt, load_template

        if settings.court_calibration:
            try:
                ManualCourt(settings.court_calibration, settings)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"court_calibration: {exc}")
        else:
            packages.add("ultralytics")
            if not settings.court_weights:
                packages.add("huggingface_hub")
            else:
                files.append((Path(settings.court_weights), "court_weights"))
            try:
                load_template(settings.court_template)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"court_template: {exc}")
            if settings.court_template is None:
                warnings.append(
                    "court_template is unset: M3 exports 30 keypoints but no homography. "
                    "Supply verified correspondences or court_calibration for metre coordinates."
                )
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
        item.add_argument("--modules", nargs="+", choices=["1", "2", "3"], default=list(settings.modules))
        if command == "run":
            item.add_argument("video", type=Path, nargs="?", default=settings.video)
            item.add_argument("--output", type=Path, default=settings.output, help="New output directory")
    inspect = sub.add_parser("court-frame", help="Export numbered court keypoints for calibration")
    inspect.add_argument("video", type=Path, nargs="?", default=settings.video)
    inspect.add_argument("--frame", type=int, default=0)
    inspect.add_argument("--output", type=Path, required=True, help="New output directory")
    return parser


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
        args = build_parser(settings).parse_args(argv)
        if args.command == "court-frame":
            court_frame(settings, args)
            print(f"Court inspection saved to {args.output.resolve()}")
            return 0
        modules = set(args.modules)
        report = preflight(settings, modules)
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
        court = None
        if "3" in modules:
            court = (
                ManualCourt(settings.court_calibration, settings)
                if settings.court_calibration
                else CourtDetector(settings)
            )
        output = Pipeline(players, shuttle, court).run(args.video, args.output)
        print(f"Analysis saved to {output}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
