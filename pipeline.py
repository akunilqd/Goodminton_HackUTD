"""Coordinate modules without coupling their inference implementations."""

import collections
import json
import logging
import math
from dataclasses import asdict, replace
from pathlib import Path

import cv2

from module3_court.geometry import draw_mini_court, fit_detector_homography, on_court, project_point
from schemas import Court, FrameResult
from video import VideoReader

LOGGER = logging.getLogger(__name__)

_TRAIL_LEN = 8  # number of past positions to draw as fading trail


# COCO's 17 joints, as returned by RTMPose-L.
_POSE_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def render_frame(frame, result, trail, pose_threshold=0.3, court_threshold=0.5):
    """Draw inference results only after all models have seen the original frame."""
    height, width = frame.shape[:2]

    def visible(point, threshold):
        x, y, score = point
        return (all(math.isfinite(v) for v in point) and score >= threshold
                and 0 <= x < width and 0 <= y < height and (x != 0 or y != 0))

    def label(text, point, color):
        x = max(0, min(width - 1, point[0]))
        y = max(14, min(height - 1, point[1]))
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)

    if result.court is not None:
        for index, point in enumerate(result.court.keypoints):
            if visible(point, court_threshold):
                xy = (round(point[0]), round(point[1]))
                cv2.circle(frame, xy, 5, (255, 255, 0), -1, cv2.LINE_AA)
                label(f'C{index}', (xy[0] + 6, xy[1] - 6), (255, 255, 0))

    for player in result.players or []:
        color = (60, 230, 90)
        x1, y1, x2, y2 = (round(v) for v in player.bbox_xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        identity = f'ID {player.track_id}' if player.track_id is not None else 'Person'
        label(f'{identity} {player.confidence:.2f} | RTMPose-L', (x1, y1 - 8), color)
        points = player.keypoints
        for a, b in _POSE_EDGES:
            if a < len(points) and b < len(points) and visible(points[a], pose_threshold) and visible(points[b], pose_threshold):
                cv2.line(frame, (round(points[a][0]), round(points[a][1])),
                         (round(points[b][0]), round(points[b][1])), (0, 165, 255), 2, cv2.LINE_AA)
        for point in points:
            if visible(point, pose_threshold):
                cv2.circle(frame, (round(point[0]), round(point[1])), 3,
                           (255, 80, 255), -1, cv2.LINE_AA)

    sample = result.shuttle
    if sample is not None and sample.visible and sample.pixel:
        trail.append(sample.pixel)
    else:
        trail.clear()
    for i, (x, y) in enumerate(trail):
        alpha = (i + 1) / len(trail)
        cv2.circle(frame, (round(x), round(y)), max(2, round(5 * alpha)),
                   (0, round(255 * alpha), round(255 * (1 - alpha))), -1)
    if trail:
        x, y = trail[-1]
        cv2.circle(frame, (round(x), round(y)), 7, (0, 255, 255), 2)
    return frame


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Pipeline:
    """Instances are injectable for testing; None explicitly disables a module."""

    def __init__(self, players=None, shuttle=None, court=None):
        if players is None and shuttle is None and court is None:
            raise ValueError("Select at least one module")
        self.players = players
        self.shuttle = shuttle
        self.court = court

    def _calibrate(self, video, confidence):
        """Read ahead for the first valid H before writing any output frames."""
        init_frame = getattr(self.court, "init_frame", 0)
        reset = getattr(self.court, "reset", None)
        if reset is not None:
            reset()
        with VideoReader(video) as reader:
            for index, _, frame in reader:
                if index < init_frame:
                    continue
                court = self.court.process(frame)
                if court is None:
                    continue
                if court.homography is not None:
                    return index, court
                try:
                    matrix, inliers, error = fit_detector_homography(court.keypoints, confidence)
                except ValueError:
                    continue
                return index, replace(
                    court, homography=matrix, inlier_count=inliers,
                    reprojection_error_m=error, status="fixed_homography",
                )
        raise ValueError(f"No frame yielded a valid homography from frame {init_frame} onward")

    def run(self, video: Path, output_dir: Path) -> Path:
        """Write the final video directly. Tracking and projection data stay in memory."""
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        video_path = output_dir / "annotated.mp4"
        try:
            pose_threshold = getattr(getattr(self.players, "settings", None), "pose_confidence", 0.3)
            court_threshold = getattr(getattr(self.court, "settings", None), "court_keypoint_confidence", 0.5)
            calibration = self._calibrate(video, court_threshold) if self.court is not None else None
            if calibration is not None:
                LOGGER.info("Fixed H calibrated from frame %s", calibration[0])

            samples = None
            if self.shuttle is not None:
                LOGGER.info("Running TrackNet and InpaintNet")
                samples = self.shuttle.predict_video(video)
            if self.players is not None:
                self.players.reset()

            with VideoReader(video) as reader:
                self._write_video(reader, video_path, calibration, samples, pose_threshold, court_threshold)
        except BaseException:
            try:
                video_path.unlink(missing_ok=True)
                output_dir.rmdir()
            except OSError:
                LOGGER.exception("Could not clean failed output %s", output_dir)
            raise
        return video_path

    def _write_video(self, reader, video_path, calibration, samples, pose_threshold, court_threshold):
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), reader.info.fps,
            (reader.info.width, reader.info.height),
        )
        trail = collections.deque(maxlen=_TRAIL_LEN)
        reference_index, reference = calibration if calibration is not None else (None, None)
        matrix = reference.homography if reference is not None else None
        count = 0
        try:
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open video writer: {video_path}")
            for index, timestamp, frame in reader:
                result = FrameResult(index, timestamp)
                if self.court is not None:
                    court = reference if index == reference_index else self.court.process(frame)
                    result.court = replace(court or Court(), homography=matrix)
                if self.players is not None:
                    result.players = self.players.process(frame, court=result.court)
                    for player in result.players:
                        player.court_xy_m = project_point(matrix, player.foot_pixel)
                        player.on_court = on_court(player.court_xy_m)
                if samples is not None:
                    if index not in samples:
                        raise ValueError(f"Shuttle output missing decoded video frame {index}")
                    result.shuttle = samples[index]
                    result.shuttle.court_projection_m = project_point(matrix, result.shuttle.pixel)

                render_frame(frame, result, trail, pose_threshold, court_threshold)
                if result.court is not None:
                    draw_mini_court(frame, asdict(result))
                writer.write(frame)
                count += 1
                if count % 100 == 0:
                    LOGGER.info("Processed %s frames", count)

            if count == 0:
                raise ValueError("Video contains no decodable frames")
            if samples is not None and len(samples) != count:
                raise ValueError("Shuttle output frame count differs from decoded video")
        finally:
            writer.release()
