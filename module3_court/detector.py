"""M3: 30-keypoint YOLO court detections and validated court-plane mapping."""

import json
from pathlib import Path

import numpy as np
import torch

from config import Settings, require_file
from schemas import Court

from .geometry import fit_homography


def load_template(path: Path | None):
    if path is None:
        return None
    with require_file(path, "court_template").open(encoding="utf-8") as stream:
        data = json.load(stream)
    points = np.asarray(data["court_points_m"], dtype=float)
    if points.shape != (30, 2) or not np.isfinite(points).all():
        raise ValueError("court_template must contain 30 finite [x,y] court_points_m")
    if len(np.unique(points, axis=0)) != 30:
        raise ValueError("Court template contains duplicate positions")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("Court template positions must not be collinear")
    return points


class CourtDetector:
    def __init__(self, settings: Settings, model=None):
        self.settings = settings
        device = torch.device(settings.device)
        if device.type == "cuda":
            device = torch.device("cuda", device.index or 0)
            torch.cuda.set_device(device)
            torch.backends.cudnn.benchmark = True
        self.device = str(device)
        from ultralytics.cfg import DEFAULT_CFG_DICT

        half = settings.gpu_half and device.type == "cuda"
        # Support current and older Ultralytics precision arguments.
        self.precision = (
            {"quantize": 16 if half else 32}
            if "quantize" in DEFAULT_CFG_DICT else {"half": half}
        )
        self.template = load_template(settings.court_template)
        if model is None:
            from ultralytics import YOLO

            weights = settings.court_weights
            if not weights:
                from huggingface_hub import hf_hub_download

                weights = hf_hub_download("Huydinh1205/court_yolo11n_pose", "court_yolo11n_pose.pt")
            model = YOLO(weights, task="pose")
        self.model = model

    def process(self, frame: np.ndarray) -> Court:
        result = self.model.predict(
            frame,
            imgsz=640,
            conf=self.settings.court_confidence,
            device=self.device,
            **self.precision,
            verbose=False,
        )[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return Court()
        index = int(result.boxes.conf.argmax().item())
        xy = result.keypoints.xy[index].cpu().numpy()
        if result.keypoints.conf is None:
            raise ValueError("Court model must provide keypoint confidence scores")
        confidence = result.keypoints.conf[index].cpu().numpy()
        if xy.shape != (30, 2) or confidence.shape != (30,):
            raise ValueError("Expected the court model's 30 keypoints; check court_weights")
        if not np.isfinite(xy).all() or not np.isfinite(confidence).all():
            return Court(status="invalid_keypoints")
        court = Court(keypoints=[tuple(p) for p in np.column_stack((xy, confidence)).tolist()])
        if self.template is None:
            court.status = "template_required"
            return court
        valid = confidence >= self.settings.court_keypoint_confidence
        # Ultralytics can zero out invisible keypoint coordinates.
        valid &= np.any(xy != 0, axis=1)
        if valid.sum() < 4:
            court.status = "insufficient_keypoints"
            return court
        try:
            court.homography, court.inlier_count, court.reprojection_error_m = fit_homography(
                xy[valid],
                self.template[valid],
                self.settings.court_ransac_threshold_m,
                self.settings.court_max_error_m,
            )
            court.status = "calibrated"
        except ValueError:
            court.status = "invalid_geometry"
        return court


class ManualCourt:
    """Explicit fixed-camera calibration; never silently reused at another resolution."""

    def __init__(self, path: Path, settings: Settings):
        with require_file(path, "court_calibration").open(encoding="utf-8") as stream:
            data = json.load(stream)
        self.image_size = tuple(data["image_size"])
        if len(self.image_size) != 2 or any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in self.image_size
        ):
            raise ValueError("Calibration image_size must be [width,height] in positive pixels")
        points = np.asarray(data["image_points"], dtype=float)
        matrix, count, error = fit_homography(
            points,
            data["court_points_m"],
            settings.court_ransac_threshold_m,
            settings.court_max_error_m,
        )
        self.court = Court(
            keypoints=[(float(x), float(y), 1.0) for x, y in points],
            homography=matrix,
            inlier_count=count,
            reprojection_error_m=error,
            status="manual_calibration",
        )

    def process(self, frame: np.ndarray) -> Court:
        if (frame.shape[1], frame.shape[0]) != self.image_size:
            raise ValueError("Video resolution differs from manual calibration image_size")
        return self.court
