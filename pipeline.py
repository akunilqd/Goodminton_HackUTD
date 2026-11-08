"""Coordinate modules without coupling their inference implementations."""

import collections
import json
import logging
import math
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

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
        if result.court.homography is not None:
            inverse = np.linalg.inv(np.asarray(result.court.homography))
            # Court-floor lines only; net height is not projected onto the floor.
            lines = [((0, y), (6.10, y)) for y in (0, .76, 4.72, 8.68, 12.64, 13.40)]
            lines += [((x, 0), (x, 13.40)) for x in (0, .46, 5.64, 6.10)]
            lines += [((3.05, 0), (3.05, 4.72)), ((3.05, 8.68), (3.05, 13.40))]
            for a, b in lines:
                pa, pb = project_point(inverse, a), project_point(inverse, b)
                if pa is not None and pb is not None:
                    cv2.line(frame, tuple(round(v) for v in pa), tuple(round(v) for v in pb),
                             (255, 255, 0), 2, cv2.LINE_AA)
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


def annotate_hits(video: Path, hits_path: Path, output: Path, hold_s: float = 0.16):
    """Render JSONL hit timestamps without rerunning inference. Output has no audio.

    Red marks the exact contact frame; amber holds its label briefly afterwards.
    Frame index and timestamp must agree with the input video's CFR timebase.
    """
    if not math.isfinite(hold_s) or hold_s < 0:
        raise ValueError("hold_s must be non-negative and finite")
    output = Path(output).resolve()
    partial = output.with_name(output.stem + ".partial.mp4")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Hit review output must be .mp4")
    if output.exists() or partial.exists():
        raise ValueError("Review output already exists; select a new filename")
    with Path(hits_path).open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    with VideoReader(video) as reader:
        fps, width, height = reader.info.fps, reader.info.width, reader.info.height
        by_frame = {}
        for event in events:
            index, timestamp = event["frame_index"], event["timestamp_s"]
            if (type(index) is not int or index < 0 or index in by_frame
                    or not math.isfinite(timestamp)
                    or not math.isclose(index / fps, timestamp, abs_tol=1e-5)):
                raise ValueError("Hit frames must be unique, non-negative and match video timestamps")
            by_frame[index] = event
        ordered = sorted(by_frame)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot write review video: {partial}")
        scale = max(.4, width / 1920)
        margin, line_height = round(20 * scale), round(36 * scale)
        hold_frames = round(hold_s * fps)
        current, total, seen = None, 0, 0
        try:
            for index, timestamp, frame in reader:
                exact = index in by_frame
                if exact:
                    current = by_frame[index]
                    seen += 1
                active = current is not None and index - current["frame_index"] <= hold_frames
                color = (0, 0, 255) if exact else (0, 190, 255)
                x = max(margin, width - round(720 * scale))
                if active:
                    panel = frame[margin:margin + 4 * line_height, x:width - margin]
                    panel[:] = (panel.astype(np.float32) * .2).astype(np.uint8)
                    player = current.get("hitter_id")
                    text = [
                        f"{'HIT NOW' if exact else 'HIT REPLAY'}  #{current['event_id']}  Player {player if player is not None else '?'}",
                        f"JSON: {current['timestamp_s']:.3f}s | frame {current['frame_index']}",
                        f"Score: {current['hit_score']:.2f} (heuristic)",
                    ]
                    for row, label in enumerate(text):
                        cv2.putText(frame, label, (x + margin, margin + (row + 1) * line_height),
                                    cv2.FONT_HERSHEY_SIMPLEX, .8 * scale, color, max(1, round(2*scale)), cv2.LINE_AA)
                if exact:
                    cv2.rectangle(frame, (2, 2), (width - 3, height - 3), color, max(2, round(8*scale)))
                # Persistent clock and counter make frame-by-frame verification easy.
                label = f"VIDEO {timestamp:.3f}s | frame {index} | hits {seen}/{len(events)}"
                y = height - margin - line_height
                cv2.putText(frame, label, (margin, y), cv2.FONT_HERSHEY_SIMPLEX,
                            .75*scale, (0, 0, 0), max(3, round(5*scale)), cv2.LINE_AA)
                cv2.putText(frame, label, (margin, y), cv2.FONT_HERSHEY_SIMPLEX,
                            .75*scale, (255, 255, 255), max(1, round(2*scale)), cv2.LINE_AA)
                denominator = max(reader.info.reported_frame_count - 1, max(ordered, default=0), 1)
                def marker(f):
                    return margin + round((width - 2*margin) * min(f / denominator, 1.))
                baseline = height - margin
                cv2.line(frame, (margin, baseline), (width-margin, baseline), (80, 80, 80), max(2, round(4*scale)))
                for hit_frame in ordered:
                    px = marker(hit_frame)
                    cv2.line(frame, (px, baseline-8), (px, baseline+8), (0, 190, 255), 2)
                cv2.circle(frame, (marker(index), baseline), max(3, round(6*scale)), (255, 255, 255), -1)
                writer.write(frame)
                total += 1
            if total == 0 or seen != len(events):
                raise ValueError("Video is empty or hit JSON contains events beyond decoded video")
        finally:
            writer.release()
        partial.replace(output)
    return {"output": str(output), "frames": total, "hits": seen, "fps": fps,
            "exact_marker": "red border", "label_hold_seconds": hold_s, "audio": False}


class Pipeline:
    """Instances are injectable for testing; None explicitly disables a module."""

    def __init__(self, players=None, shuttle=None, court=None, hits=None):
        if players is None and shuttle is None and court is None:
            raise ValueError("Select at least one module")
        self.players = players
        self.shuttle = shuttle
        self.court = court
        self.hits = hits
        if hits is not None and (players is None or shuttle is None):
            raise ValueError("Module 4 requires player and shuttle modules")

    def _calibrate(self, video, confidence):
        """Read ahead for the first valid H before writing any output frames."""
        init_frame = getattr(self.court, "init_frame", 0)
        reset = getattr(self.court, "reset", None)
        if reset is not None:
            reset()
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
                    "m4": self.hits is not None,
                },
                "processed_frames": 0,
                "court_coordinates": "metres: far-left origin, x across 6.10m, y toward near baseline 13.40m",
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
                    self.players.reset()
                if self.court is not None:
                    if hasattr(self.court, "reset"):
                        self.court.reset()
                    init_index = getattr(getattr(self.court, "settings", None), "court_init_frame", 0)
                    with VideoReader(video) as calibration_reader:
                        for frame_index, _, calibration_frame in calibration_reader:
                            if frame_index == init_index:
                                self.court.process(calibration_frame)
                                break
                        else:
                            raise ValueError(f"Video has no court initialization frame {init_index}")
                    metadata["court_calibration"] = {
                        "mode": "fixed", "frame_index": init_index,
                        "corner_order": ["TL", "TR", "BR", "BL"],
                        "fit_error_is_independent_accuracy": False,
                    }
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

                if self.hits is not None:
                    from module4 import write_hits

                    LOGGER.info("Detecting hits from aligned M1/M2 records on CPU")
                    with partial.open(encoding="utf-8") as stream:
                        events, summary = self.hits.detect(
                            (json.loads(line) for line in stream), reader.info.fps,
                            reader.info.width, reader.info.height,
                        )
                    write_hits(output_dir, events, summary)
                    metadata["hit_detection"] = summary

                writer.release()
                writer = None
                LOGGER.info("Annotated video written to %s", video_path)
                partial.replace(output_dir / "frames.jsonl")
                if self.court is not None:
                    metadata["court_model_inferences"] = getattr(self.court, "inference_count", 0)
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
