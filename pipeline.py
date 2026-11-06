"""Coordinate modules without coupling their inference implementations."""

import collections
import csv
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import cv2

from module3_court.geometry import on_court, project_point
from schemas import FrameResult
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

    def run(self, video: Path, output_dir: Path) -> Path:
        # Open before creating artifacts; never overwrite an earlier run.
        with VideoReader(video) as reader:
            output_dir = output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=False)
            metadata = {
                "schema_version": "1.0",
                "status": "running",
                "video": asdict(reader.info),
                "modules": {
                    "m1": self.players is not None,
                    "m2": self.shuttle is not None,
                    "m3": self.court is not None,
                },
                "processed_frames": 0,
                "court_coordinates": "metres: x across 6.10m, y along 13.40m; template origin",
                "shuttle_projection": "court-plane projection; airborne 3D position is unknown",
            }
            metadata_path = output_dir / "metadata.json"
            write_json(metadata_path, metadata)
            partial = output_dir / "frames.partial.jsonl"
            count = 0
            writer = None
            video_path = output_dir / ("shuttle.mp4" if self.shuttle is not None else "annotated.mp4")
            trail = collections.deque(maxlen=_TRAIL_LEN)
            pose_threshold = getattr(getattr(self.players, "settings", None), "pose_confidence", 0.3)
            court_threshold = getattr(getattr(self.court, "settings", None), "court_keypoint_confidence", 0.5)
            try:
                if self.players is not None:
                    self.players.reset()
                samples = None
                if self.shuttle is not None:
                    LOGGER.info("Running TrackNet and InpaintNet passes")
                    samples = self.shuttle.predict_video(video, output_dir / "shuttle")
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), reader.info.fps,
                    (reader.info.width, reader.info.height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open video writer: {video_path}")
                with partial.open("w", encoding="utf-8") as stream:
                    for index, timestamp, frame in reader:
                        result = FrameResult(index, timestamp)
                        if self.court is not None:
                            result.court = self.court.process(frame)
                        matrix = result.court.homography if result.court is not None else None
                        if self.players is not None:
                            result.players = self.players.process(frame, court=result.court)
                            for player in result.players:
                                player.court_xy_m = project_point(matrix, player.foot_pixel)
                                player.on_court = on_court(player.court_xy_m)
                        if samples is not None:
                            if index not in samples:
                                raise ValueError(
                                    f"Shuttle output missing decoded video frame {index}"
                                )
                            result.shuttle = samples[index]
                            result.shuttle.court_projection_m = project_point(
                                matrix, result.shuttle.pixel
                            )
                        stream.write(json.dumps(asdict(result), allow_nan=False) + "\n")
                        writer.write(render_frame(frame, result, trail, pose_threshold, court_threshold))
                        count += 1
                        if count % 100 == 0:
                            LOGGER.info("Processed %s frames", count)
                if count == 0:
                    raise ValueError("Video contains no decodable frames")
                if samples is not None and set(samples) != set(range(count)):
                    raise ValueError("Shuttle output frame count differs from decoded video")
                if samples is not None:
                    self._write_shuttle_csv(output_dir / "shuttle.csv", samples, reader.info.fps)

                writer.release()
                writer = None
                LOGGER.info("Annotated video written to %s", video_path)
                partial.replace(output_dir / "frames.jsonl")
                metadata.update(status="complete", processed_frames=count)
            except BaseException as exc:
                metadata.update(status="failed", processed_frames=count, error=str(exc))
                raise
            finally:
                if writer is not None:
                    writer.release()
                write_json(metadata_path, metadata)
        return output_dir

    @staticmethod
    def _write_shuttle_csv(path, samples, fps):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "frame_index",
                    "timestamp_s",
                    "x",
                    "y",
                    "visible",
                    "confidence",
                    "raw_x",
                    "raw_y",
                    "rectified",
                    "source",
                    "court_x_m",
                    "court_y_m",
                ]
            )
            for index, sample in sorted(samples.items()):
                writer.writerow(
                    [
                        index,
                        index / fps,
                        *(sample.pixel or (None, None)),
                        int(sample.visible),
                        sample.confidence,
                        *(sample.raw_pixel or (None, None)),
                        int(sample.rectified),
                        sample.source,
                        *(sample.court_projection_m or (None, None)),
                    ]
                )
