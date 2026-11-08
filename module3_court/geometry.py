"""M3 geometry and standalone mini-court rendering from detector output."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
# CourtKeyNet returns the four corners in TL, TR, BR, BL order.
COURT_CORNERS_M = (
    (0.0, 0.0),
    (COURT_WIDTH_M, 0.0),
    (COURT_WIDTH_M, COURT_LENGTH_M),
    (0.0, COURT_LENGTH_M),
)
SINGLES_INSET_M = 0.46
LONG_SERVICE_INSET_M = 0.76
SHORT_SERVICE_FROM_NET_M = 1.98

# Indices refer to this project's 30-point detector. Use the floor landmarks
# identifiable in the numbered reference frame and omit ambiguous detections.
_CENTER_X = COURT_WIDTH_M / 2
_NET_Y = COURT_LENGTH_M / 2
_RIGHT_SINGLES_X = COURT_WIDTH_M - SINGLES_INSET_M
_FAR_SHORT_Y = _NET_Y - SHORT_SERVICE_FROM_NET_M
_NEAR_SHORT_Y = _NET_Y + SHORT_SERVICE_FROM_NET_M
_NEAR_LONG_Y = COURT_LENGTH_M - LONG_SERVICE_INSET_M
DETECTOR_LANDMARKS_M = {
    2: (_CENTER_X, 0.0),
    3: (SINGLES_INSET_M, 0.0),
    5: (COURT_WIDTH_M, LONG_SERVICE_INSET_M),
    7: (_CENTER_X, LONG_SERVICE_INSET_M),
    9: (0.0, LONG_SERVICE_INSET_M),
    10: (0.0, _FAR_SHORT_Y),
    11: (SINGLES_INSET_M, _FAR_SHORT_Y),
    12: (_CENTER_X, _FAR_SHORT_Y),
    13: (_RIGHT_SINGLES_X, _FAR_SHORT_Y),
    14: (COURT_WIDTH_M, _FAR_SHORT_Y),
    15: (0.0, _NEAR_SHORT_Y),
    16: (SINGLES_INSET_M, _NEAR_SHORT_Y),
    17: (_CENTER_X, _NEAR_SHORT_Y),
    18: (_RIGHT_SINGLES_X, _NEAR_SHORT_Y),
    20: (COURT_WIDTH_M, _NEAR_LONG_Y),
    21: (_RIGHT_SINGLES_X, _NEAR_LONG_Y),
    22: (_CENTER_X, _NEAR_LONG_Y),
    24: (0.0, _NEAR_LONG_Y),
    25: (0.0, COURT_LENGTH_M),
    26: (SINGLES_INSET_M, COURT_LENGTH_M),
    27: (_CENTER_X, COURT_LENGTH_M),
    28: (_RIGHT_SINGLES_X, COURT_LENGTH_M),
    29: (COURT_WIDTH_M, COURT_LENGTH_M),
}


def project_point(homography, point):
    if homography is None or point is None:
        return None
    matrix = np.asarray(homography, dtype=float)
    pixel = np.asarray([*point, 1.0], dtype=float)
    mapped = matrix @ pixel
    if not np.isfinite(mapped).all() or abs(mapped[2]) < 1e-10:
        return None
    return (float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2]))


def on_court(point, margin_m: float = 0.0):
    if point is None:
        return None
    return (
        -margin_m <= point[0] <= COURT_WIDTH_M + margin_m
        and -margin_m <= point[1] <= COURT_LENGTH_M + margin_m
    )


def fit_homography(image_points, court_points, threshold_m=0.15, max_error_m=0.2):
    """Reject degenerate/poor fits; the RANSAC threshold is in destination metres."""
    image_points = np.asarray(image_points, dtype=np.float64)
    court_points = np.asarray(court_points, dtype=np.float64)
    if (
        image_points.ndim != 2
        or image_points.shape[1] != 2
        or image_points.shape != court_points.shape
        or len(image_points) < 4
    ):
        raise ValueError("Homography requires at least four paired 2D points")
    if not np.isfinite(image_points).all() or not np.isfinite(court_points).all():
        raise ValueError("Homography points must be finite")
    for points in (image_points, court_points):
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("Homography points must not be collinear")
    matrix, mask = cv2.findHomography(image_points, court_points, cv2.RANSAC, threshold_m)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        raise ValueError("Could not estimate a valid homography")
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Homography is singular")
    inliers = mask.ravel().astype(bool)
    count = int(inliers.sum())
    if count < 4 or count / len(image_points) < 0.5:
        raise ValueError("Too few consistent court keypoints")
    projected = cv2.perspectiveTransform(image_points[inliers, None, :], matrix)[:, 0, :]
    error = float(np.sqrt(np.mean(np.sum((projected - court_points[inliers]) ** 2, axis=1))))
    if not np.isfinite(error) or error > max_error_m:
        raise ValueError(f"Court reprojection error {error:.3f} m exceeds {max_error_m} m")
    return matrix.tolist(), count, error


def fit_detector_homography(keypoints, min_confidence=0.5):
    """Fit from ordered CourtKeyNet corners or legacy YOLO landmarks."""
    points = np.asarray(keypoints, dtype=float)
    if points.shape == (4, 3):
        if not np.isfinite(points).all() or np.any(points[:, 2] < min_confidence):
            raise ValueError("All four court corners must have valid coordinates and confidence")
        corners = points[:, :2]
        contour = corners.astype(np.float32)
        if not cv2.isContourConvex(contour) or cv2.contourArea(contour, oriented=True) <= 0:
            raise ValueError("Court corners must form a convex quadrilateral in TL, TR, BR, BL order")
        return fit_homography(corners, COURT_CORNERS_M)

    if points.shape != (30, 3):
        raise ValueError("Expected four court corners or 30 legacy keypoints with x, y and confidence")
    source, destination = [], []
    for index, court_point in DETECTOR_LANDMARKS_M.items():
        x, y, confidence = points[index]
        if np.isfinite(points[index]).all() and confidence >= min_confidence and (x or y):
            source.append((x, y))
            destination.append(court_point)
    return fit_homography(source, destination)


def draw_mini_court(frame, record):
    """Draw a top-down court and positions on the upper-right of the frame."""
    height, width = frame.shape[:2]
    padding = 8
    scale = 1.5 * min(width * 0.18 / COURT_WIDTH_M, height * 0.40 / COURT_LENGTH_M)
    left = width - 2 * padding - round(COURT_WIDTH_M * scale)
    top = 2 * padding

    def to_pixel(point):
        x, y = point
        return left + round(x * scale), top + round(y * scale)

    right, bottom = to_pixel((COURT_WIDTH_M, COURT_LENGTH_M))
    panel = frame[top - padding:bottom + padding + 1,
                  max(0, left - padding):right + padding + 1]
    if panel.size:
        tint = np.full_like(panel, (175, 200, 182))
        cv2.addWeighted(panel, 0.5, tint, 0.5, 0, dst=panel)

    def line(start, end):
        cv2.line(frame, to_pixel(start), to_pixel(end), (36, 28, 24), 1, cv2.LINE_AA)

    for x in (0.0, SINGLES_INSET_M, _RIGHT_SINGLES_X, COURT_WIDTH_M):
        line((x, 0.0), (x, COURT_LENGTH_M))
    for y in (0.0, LONG_SERVICE_INSET_M, _FAR_SHORT_Y, _NET_Y,
              _NEAR_SHORT_Y, _NEAR_LONG_Y, COURT_LENGTH_M):
        line((0.0, y), (COURT_WIDTH_M, y))
    line((_CENTER_X, 0.0), (_CENTER_X, _FAR_SHORT_Y))
    line((_CENTER_X, _NEAR_SHORT_Y), (_CENTER_X, COURT_LENGTH_M))

    court = record.get("court") or {}
    matrix = court.get("homography")
    if matrix is None:
        return
    keypoints = court["keypoints"]
    for index, (x, y, confidence) in enumerate(keypoints):
        if len(keypoints) == 30 and index not in DETECTOR_LANDMARKS_M:
            continue
        point = project_point(matrix, (x, y)) if confidence >= 0.5 and (x or y) else None
        if on_court(point, margin_m=0.15):
            cv2.circle(frame, to_pixel(point), 2, (255, 220, 65), -1, cv2.LINE_AA)
    for player in record.get("players") or []:
        point = player["court_xy_m"]
        if on_court(point):
            cv2.circle(frame, to_pixel(point), 4, (255, 120, 35), -1, cv2.LINE_AA)


def find_fixed_homography(detections):
    """Find the first valid calibration in a detector JSONL stream."""
    for raw_record in detections:
        record = json.loads(raw_record)
        court = record.get("court") or {}
        if court.get("status") in {"fixed_courtkeynet", "manual_calibration", "fixed_homography"}:
            matrix = court.get("homography")
            if matrix is not None:
                reference = court.get("homography_reference_frame", record["frame_index"])
                return reference, (
                    matrix, court["inlier_count"], court["reprojection_error_m"]
                )
        try:
            fit = fit_detector_homography(court.get("keypoints", []))
        except ValueError:
            continue
        return record["frame_index"], fit
    raise ValueError("No frame yielded a valid homography")


def read_tracking_output(path, field):
    """Read one project's tracking output keyed by zero-based frame index."""
    samples = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if field == "shuttle" and path.suffix.lower() == ".csv":
            rows = csv.DictReader(stream)
            if not {"frame_index", "visible", "x", "y"}.issubset(rows.fieldnames or ()):
                raise ValueError(f"{path}: expected Module 2 shuttle.csv columns")
            for row in rows:
                index = int(row["frame_index"])
                visible = row["visible"] == "1"
                if index in samples or index < 0 or row["visible"] not in {"0", "1"}:
                    raise ValueError(f"{path}: invalid or duplicate frame {index}")
                pixel = [float(row["x"]), float(row["y"])] if visible else None
                samples[index] = {"pixel": pixel, "visible": visible}
        else:
            for line in stream:
                row = json.loads(line)
                index = row["frame_index"]
                if type(index) is not int or index < 0 or index in samples:
                    raise ValueError(f"{path}: invalid or duplicate frame {index}")
                samples[index] = row[field]
    if not samples or set(samples) != set(range(len(samples))):
        raise ValueError(f"{path}: expected contiguous frame indices starting at 0")
    return samples


def render_detector_output(
    video_path, detections_path, output_dir, *, fixed_homography=True,
    players_path=None, shuttle_path=None,
):
    """Create a mini-court video and enriched JSONL aligned by frame index."""
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    tracking = {}
    for field, path in (("players", players_path), ("shuttle", shuttle_path)):
        if path is not None:
            tracking[field] = read_tracking_output(path, field)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        size = tuple(int(capture.get(prop)) for prop in
                     (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0 or min(size) <= 0:
            raise ValueError("Video must have valid dimensions and frame rate")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output_dir.parent) as temp_dir:
            staged = Path(temp_dir) / "result"
            staged.mkdir()
            writer = cv2.VideoWriter(str(staged / "annotated.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            count = calibrated = 0
            try:
                if not writer.isOpened():
                    raise ValueError("Cannot create annotated video")
                with detections_path.open(encoding="utf-8") as detections, (
                    staged / "frames.jsonl"
                ).open("w", encoding="utf-8") as output:
                    if fixed_homography:
                        reference_frame, fixed_fit = find_fixed_homography(detections)
                        detections.seek(0)
                    for index, raw_record in enumerate(detections):
                        record = json.loads(raw_record)
                        ok, frame = capture.read()
                        if not ok or record["frame_index"] != index:
                            raise ValueError(f"Video and detections diverge at frame {index}")
                        for field, samples in tracking.items():
                            if index not in samples:
                                raise ValueError(f"{field} output is missing frame {index}")
                            record[field] = samples[index]
                        court = record.get("court")
                        matrix = None
                        if fixed_homography:
                            matrix, inliers, error = fixed_fit
                            if court is None:
                                court = record["court"] = {"keypoints": []}
                            court.update(homography=matrix, inlier_count=inliers,
                                         reprojection_error_m=error, status="fixed_homography",
                                         homography_reference_frame=reference_frame)
                        elif court is not None:
                            try:
                                matrix, inliers, error = fit_detector_homography(court["keypoints"])
                            except ValueError:
                                inliers, error = 0, None
                            court.update(homography=matrix, inlier_count=inliers,
                                         reprojection_error_m=error,
                                         status="calibrated" if matrix else "invalid_geometry")
                        for player in record.get("players") or []:
                            point = project_point(matrix, player["foot_pixel"])
                            player.update(court_xy_m=point, on_court=on_court(point))
                        if record.get("shuttle") is not None:
                            shuttle = record["shuttle"]
                            shuttle["court_projection_m"] = project_point(matrix, shuttle["pixel"])
                        draw_mini_court(frame, record)
                        writer.write(frame)
                        output.write(json.dumps(record, allow_nan=False) + "\n")
                        count += 1
                        calibrated += matrix is not None
                if capture.read()[0]:
                    raise ValueError("Video has more frames than the detections file")
                for field, samples in tracking.items():
                    if len(samples) != count:
                        raise ValueError(f"{field} output has more frames than the video")
                if calibrated == 0:
                    raise ValueError("No frame yielded a valid homography")
            finally:
                writer.release()
            staged.rename(output_dir)
    finally:
        capture.release()
    return count, calibrated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a 2D mini-court using Module 3 detections")
    parser.add_argument("video", type=Path, help="Video matching the detections")
    parser.add_argument("detections", type=Path, help="Module 3 frames.jsonl")
    parser.add_argument("--players", type=Path, help="Module 1 frames.jsonl for the same video")
    parser.add_argument("--shuttle", type=Path, help="Module 2 frames.jsonl or shuttle.csv")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixed-homography", dest="fixed_homography", action="store_true",
                      help="Reuse the detector's fixed H or the first valid fit (default)")
    mode.add_argument("--per-frame", dest="fixed_homography", action="store_false",
                      help="Fit H on each frame using detector keypoints")
    parser.set_defaults(fixed_homography=True)
    args = parser.parse_args()
    try:
        total, mapped = render_detector_output(
            args.video, args.detections, args.output, fixed_homography=args.fixed_homography,
            players_path=args.players, shuttle_path=args.shuttle,
        )
    except (OSError, ValueError, KeyError, cv2.error) as exc:
        parser.exit(1, f"Error: {exc}\n")
    if args.fixed_homography:
        print(f"One homography applied to {mapped}/{total} frames. Saved to {args.output}")
    else:
        print(f"Homography computed for {mapped}/{total} frames. Saved to {args.output}")
