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
    if "5" in modules and "4" not in modules:
        errors.append("Module 5 requires Module 4 (--modules 1 2 3 4 5).")
    if "1" in modules and "3" not in modules:
        errors.append("Two-player court tracking requires Module 3; use --modules 1 3 (or 1 2 3).")
    packages = set()
    files = []
    devices = {}
    uses_court_model = "3" in modules and not (settings.court_calibration or use_saved_court)
    if "1" in modules or "5" in modules or uses_court_model:
        try:
            device = configure_device(settings.device)
            for module in modules & {"1", "3", "5"}:
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
    if "5" in modules:
        packages.update(("torch", "positional_encodings", "torchinfo"))
        bst_repo = Path(settings.bst_repo) / "stroke_classification"
        if not bst_repo.is_dir():
            errors.append(
                f"BST repo not found at {settings.bst_repo}. "
                "Clone: git clone https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer "
                + str(settings.bst_repo)
            )
        if not Path(settings.bst_weights).is_file():
            errors.append(
                f"BST weights not found at {settings.bst_weights}. "
                "Download from: https://drive.google.com/file/d/1KXY62Q-5kSkgqUiJY88DApdsI2qDqZC5/view"
            )
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
    parser = argparse.ArgumentParser(description="Badminton analysis pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "check"):
        item = sub.add_parser(command)
        item.add_argument("--modules", nargs="+", choices=["1", "2", "3", "4", "5"], default=list(settings.modules))
        if command == "run":
            item.add_argument("video", type=Path, nargs="?", default=settings.video)
            item.add_argument("--court-init-frame", type=int, default=settings.court_init_frame)
            item.add_argument("--output", type=Path, default=settings.output, help="New output directory")
            item.add_argument("--court-detections", type=Path, default=None,
                              help="frames.jsonl from a prior run; reuse its fixed court calibration")
            item.add_argument("--stdb", action="store_true",
                              help="Push results to SpacetimeDB after each stage")
            item.add_argument("--stdb-module", default="badminton", help="SpacetimeDB module name")
            item.add_argument("--stdb-server", default="https://maincloud.spacetimedb.com",
                              help="SpacetimeDB server URL")
    inspect = sub.add_parser("court-frame", help="Export numbered court keypoints for calibration")
    inspect.add_argument("video", type=Path, nargs="?", default=settings.video)
    inspect.add_argument("--frame", type=int, default=0)
    inspect.add_argument("--output", type=Path, required=True, help="New output directory")
    hits = sub.add_parser("detect-hits", help="Run CPU-only M4 on a complete saved M1/M2 run")
    hits.add_argument("run_dir", type=Path)
    hits.add_argument("--output", type=Path, required=True, help="New output directory")
    hits.add_argument("--rallies", type=Path, help="Optional JSON list of inclusive rally frame ranges")
    classify = sub.add_parser("classify-shots", help="Run BadmintonDB BST (18 to 6 classes) on a completed M1-M4 run")
    classify.add_argument("run_dir", type=Path)
    classify.add_argument("--output", type=Path, required=True, help="New output directory")
    review = sub.add_parser("annotate-hits", help="Mark hit JSONL timestamps on a video without inference")
    review.add_argument("video", type=Path)
    review.add_argument("--hits", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True, help="New .mp4 output file")
    review.add_argument("--hold-seconds", type=float, default=0.16)
    ann_shots = sub.add_parser("annotate-shots", help="Overlay BST shot labels on a run's annotated video")
    ann_shots.add_argument("run_dir", type=Path, help="Run directory containing shuttle.mp4 and shots.jsonl")
    ann_shots.add_argument("--output", type=Path, required=True, help="Output .mp4 file path")
    speed_cmd = sub.add_parser("speed-csv", help="Export per-frame player and shuttle speed to CSV")
    speed_cmd.add_argument("run_dir", type=Path, help="Run directory containing frames.jsonl")
    speed_cmd.add_argument("--output", type=Path, required=True, help="Output .csv file path")
    summary_cmd = sub.add_parser("summary-report", help="Generate HTML match summary with charts and heatmaps")
    summary_cmd.add_argument("run_dir", type=Path, help="Run directory containing frames.jsonl, shots.jsonl, speed.csv")
    summary_cmd.add_argument("--output", type=Path, required=True, help="Output .html file path")
    stdb_cmd = sub.add_parser("stdb-upload", help="Upload run data to SpacetimeDB via HTTP REST")
    stdb_cmd.add_argument("run_dir", type=Path, help="Run directory to upload")
    stdb_cmd.add_argument("--module", default="badminton", help="SpacetimeDB module name (default: badminton)")
    stdb_cmd.add_argument("--server", default="http://localhost:3000", help="SpacetimeDB server URL")
    stdb_cmd.add_argument("--token", default=None, help="Auth token (optional)")
    stdb_cmd.add_argument("--no-clear", action="store_true", help="Skip clearing existing data for this run_id")
    rallies_cmd = sub.add_parser("publish-rallies", help="Build rallies for a saved run and publish them to SpacetimeDB")
    rallies_cmd.add_argument("run_dir", type=Path, help="Run directory containing frames.jsonl and hits.jsonl")
    rallies_cmd.add_argument("--module", default="badminton", help="SpacetimeDB module name (default: badminton)")
    rallies_cmd.add_argument("--server", default="https://maincloud.spacetimedb.com", help="SpacetimeDB server URL")
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
        if args.command == "annotate-shots":
            from pipeline import annotate_shots

            result = annotate_shots(args.run_dir, args.output)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "detect-hits":
            from module4 import detect_saved_run

            summary = detect_saved_run(settings, args.run_dir, args.output, args.rallies)
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "classify-shots":
            from module5_shots import classify_saved_run

            summary = classify_saved_run(settings, args.run_dir, args.output)
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "speed-csv":
            from module6_analytics import export_speed_csv

            result = export_speed_csv(args.run_dir, args.output)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "summary-report":
            from module7_summary import generate_report

            result = generate_report(args.run_dir, args.output)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "stdb-upload":
            from module7_summary import upload_run

            result = upload_run(
                args.run_dir,
                module=args.module,
                server=args.server,
                token=args.token,
                clear_first=not args.no_clear,
            )
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "publish-rallies":
            from module7_summary.rallies import publish_saved_run

            print(json.dumps(publish_saved_run(args.run_dir, module=args.module, server=args.server), indent=2))
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

        shot_classifier = None
        if "5" in modules:
            from module5_shots import ShotClassifier
            shot_classifier = ShotClassifier(settings)

        stdb = None
        if getattr(args, "stdb", False):
            from stdb_client import StdbClient
            stdb = StdbClient(module=args.stdb_module, server=args.stdb_server)
        output = Pipeline(players, shuttle, court, hits, shot_classifier).run(args.video, args.output, stdb=stdb)
        print(f"Analysis saved to {output}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
