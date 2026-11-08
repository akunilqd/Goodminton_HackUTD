"""Coordinate modules without coupling their inference implementations."""

import collections
import csv
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


def annotate_shots(run_dir: Path, output: Path):
    """Overlay BST shot labels on the existing annotated video from a run directory."""
    run_dir = Path(run_dir).resolve()
    output  = Path(output).resolve()
    partial = output.with_name(output.stem + ".partial.mp4")

    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must be .mp4")
    if output.exists() or partial.exists():
        raise ValueError("Output already exists; select a new filename")

    # Load shots
    shots_path = run_dir / "shots.jsonl"
    if not shots_path.exists():
        raise ValueError(f"shots.jsonl not found in {run_dir}")
    shots = []
    with shots_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                shots.append(json.loads(line))

    # Build frame → shot lookup
    frame_shot: dict[int, dict] = {}
    for s in shots:
        for fi in range(s["start_frame"], s["end_frame"] + 1):
            frame_shot[fi] = s

    # A standalone M5 replay references the M1-M4 run instead of copying video.
    candidates = [run_dir / "shuttle.mp4", run_dir / "annotated.mp4"]
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_run"):
            source_dir = Path(metadata["source_run"])
            candidates.extend([source_dir / "shuttle.mp4", source_dir / "annotated.mp4"])
        source_video = metadata.get("video", {}).get("source")
        if source_video:
            candidates.append(Path(source_video))
    video_path = next((path for path in candidates if path.is_file()), None)
    if video_path is None:
        raise ValueError(f"No source video found for {run_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write video: {partial}")

    # Side colors, with yellow for low-confidence predictions (still labeled).
    _COLORS = {
        "Top":    (220, 100,  40),   # BGR: deep blue
        "Bottom": ( 30, 140, 230),   # BGR: orange
        None:     (110, 110, 110),   # BGR: gray
    }

    scale = max(0.5, width / 1920)
    pad   = round(14 * scale)
    lh    = round(34 * scale)   # line height
    fs    = 0.75 * scale        # font scale

    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            shot = frame_shot.get(index)
            if shot is not None:
                side  = shot.get("player_side")
                label = shot["shot_type_en"]
                conf  = shot["confidence"]
                pid   = shot.get("hitter_id")
                color = _COLORS.get(side, _COLORS[None])
                if shot.get("low_confidence", False):
                    color = (0, 230, 255)

                # Banner background (top-left)
                bw = round(400 * scale)
                bh = lh * 3 + pad * 2
                x0, y0 = pad, pad
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0, y0), (x0 + bw, y0 + bh), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

                # Colored left accent bar
                cv2.rectangle(frame, (x0, y0), (x0 + round(5 * scale), y0 + bh), color, -1)

                x_text = x0 + round(12 * scale)

                # Line 1: shot type
                is_hit_frame = index == shot["start_frame"]
                prefix = "HIT  " if is_hit_frame else "SHOT "
                cv2.putText(frame, f"{prefix}#{shot['shot_id']}  {label}",
                            (x_text, y0 + pad + lh - round(6*scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, color,
                            max(1, round(2 * scale)), cv2.LINE_AA)

                # Line 2: side + player
                side_str = side if side else "Side unavailable"
                player_str = f"P{pid}" if pid else "P?"
                low_marker = " | LOW CONF" if shot.get("low_confidence", False) else ""
                cv2.putText(frame, f"{side_str}  |  {player_str}{low_marker}",
                            (x_text, y0 + pad + lh * 2 - round(4*scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, (200, 200, 200),
                            max(1, round(1 * scale)), cv2.LINE_AA)

                # Line 3: confidence bar
                bar_w = bw - round(70 * scale)
                bar_h = round(8 * scale)
                bx = x_text
                by = y0 + pad + lh * 3 - round(10 * scale)
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (60, 60, 60), -1)
                cv2.rectangle(frame, (bx, by), (bx + round(bar_w * conf), by + bar_h), color, -1)
                cv2.putText(frame, f"{conf*100:.0f}%",
                            (bx + bar_w + round(6*scale), by + bar_h),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, (200, 200, 200),
                            1, cv2.LINE_AA)

                # Flash border on first frame of shot
                if is_hit_frame:
                    thick = max(3, round(8 * scale))
                    cv2.rectangle(frame, (thick, thick),
                                  (width - thick, height - thick), color, thick)

            # Frame counter bottom-right
            cv2.putText(frame, f"frame {index}",
                        (width - round(140 * scale), height - round(12 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (180, 180, 180),
                        1, cv2.LINE_AA)

            writer.write(frame)
            index += 1
    finally:
        cap.release()
        writer.release()

    if index == 0:
        raise ValueError("Source video has no frames")

    partial.replace(output)
    return {"output": str(output), "frames": index, "shots_annotated": len(shots), "fps": fps}


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

    def __init__(self, players=None, shuttle=None, court=None, hits=None, shot_classifier=None):
        if players is None and shuttle is None and court is None:
            raise ValueError("Select at least one module")
        self.players = players
        self.shuttle = shuttle
        self.court = court
        self.hits = hits
        self.shot_classifier = shot_classifier
        if hits is not None and (players is None or shuttle is None):
            raise ValueError("Module 4 requires player and shuttle modules")
        if shot_classifier is not None and hits is None:
            raise ValueError("Module 5 requires Module 4")
        if shot_classifier is not None and court is None:
            raise ValueError("Module 5 requires Module 3 court homography")

    def _calibrate(self, video, confidence):
        """Read the court init frame and return (frame_index, Court) with a valid H."""
        init_index = getattr(getattr(self.court, "settings", None), "court_init_frame", 0)
        reset_fn = getattr(self.court, "reset", None)
        if reset_fn is not None:
            reset_fn()
        with VideoReader(video) as reader:
            for frame_index, _, frame in reader:
                if frame_index == init_index:
                    result = self.court.process(frame)
                    if result is not None and result.homography is not None:
                        return frame_index, result
                    raise ValueError(
                        f"Court calibration failed at frame {init_index}: homography not found"
                    )
        raise ValueError(f"Video has no court initialization frame {init_index}")

    def run(self, video, output_dir, stdb=None):
        """Run the pipeline.

        stdb: optional StdbClient instance.  When provided, results are pushed
        to SpacetimeDB after each stage.  Local JSONL files are always written
        first so processing is never slowed by network latency.
        """
        video = Path(video)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=False)

        pose_threshold = getattr(getattr(self.players, "settings", None), "pose_confidence", 0.3)
        court_threshold = getattr(getattr(self.court, "settings", None), "court_keypoint_confidence", 0.5)

        with VideoReader(video) as reader:
            video_info = asdict(reader.info)

        metadata = {
            "schema_version": "1.0",
            "status": "running",
            "video": video_info,
            "modules": {
                "m1": self.players is not None,
                "m2": self.shuttle is not None,
                "m3": self.court is not None,
                "m4": self.hits is not None,
                "m5": self.shot_classifier is not None,
            },
            "processed_frames": 0,
            "court_coordinates": "metres: far-left origin, x across 6.10m, y toward near baseline 13.40m",
            "shuttle_projection": "court-plane projection; airborne 3D position is unknown",
        }
        metadata_path = output_dir / "metadata.json"
        write_json(metadata_path, metadata)

        video_path = output_dir / ("shuttle.mp4" if self.shuttle is not None else "annotated.mp4")
        partial = output_dir / "frames.partial.jsonl"
        count = 0
        run_id = output_dir.name

        try:
            calibration = None
            if self.court is not None:
                calibration = self._calibrate(video, court_threshold)
                LOGGER.info("Fixed H calibrated from frame %s", calibration[0])
                metadata["court_calibration"] = {
                    "mode": "fixed",
                    "frame_index": calibration[0],
                    "corner_order": ["TL", "TR", "BR", "BL"],
                    "fit_error_is_independent_accuracy": False,
                }

            samples = None
            if self.shuttle is not None:
                LOGGER.info("Running TrackNet and InpaintNet passes")
                samples = self.shuttle.predict_video(video)

            if self.players is not None:
                self.players.reset()

            count = self._write_video(
                video, video_path, partial, calibration, samples, pose_threshold, court_threshold
            )

            if samples is not None:
                if set(samples) != set(range(count)):
                    raise ValueError("Shuttle output frame count differs from decoded video")
                self._write_shuttle_csv(output_dir / "shuttle.csv", samples, video_info["fps"])

            # ── DB push: run_meta + running_details (after M1-M3) ─────────────
            if stdb is not None:
                try:
                    stdb.clear_run(run_id)
                    stdb.push_run_meta(
                        run_id,
                        str(video),
                        float(video_info["fps"]),
                        count,
                    )
                    frame_rows = []
                    with partial.open(encoding="utf-8") as stream:
                        for line in stream:
                            if line.strip():
                                frame_rows.append(json.loads(line))
                    LOGGER.info("Pushing %d frames to SpacetimeDB (running_details)", len(frame_rows))
                    stdb.push_running_details_batch(run_id, frame_rows)
                except Exception as exc:
                    LOGGER.warning("SpacetimeDB push (running_details) failed: %s", exc)

            events = []
            if self.hits is not None:
                from module4 import write_hits

                LOGGER.info("Detecting hits from aligned M1/M2 records on CPU")
                with partial.open(encoding="utf-8") as stream:
                    events, hit_summary = self.hits.detect(
                        (json.loads(line) for line in stream),
                        video_info["fps"], video_info["width"], video_info["height"],
                    )
                write_hits(output_dir, events, hit_summary)
                metadata["hit_detection"] = hit_summary

            shot_events = []
            if self.shot_classifier is not None and self.hits is not None:
                from module5_shots import write_shots

                LOGGER.info("Classifying shots with BST (M5)")
                frames_dict: dict[int, dict] = {}
                with partial.open(encoding="utf-8") as stream:
                    for line in stream:
                        r = json.loads(line)
                        frames_dict[r["frame_index"]] = r
                hits_list = [asdict(e) for e in events]
                shot_events = self.shot_classifier.classify(
                    hits_list, frames_dict,
                    video_info["fps"], video_info["width"], video_info["height"],
                )
                shot_summary = write_shots(
                    output_dir, shot_events, self.shot_classifier.settings,
                    self.shot_classifier.last_run_summary,
                )
                metadata["shot_classification"] = {
                    k: v for k, v in shot_summary.items() if k != "shots"
                }
                LOGGER.info("Rendering M5 labels to shots.mp4")
                annotate_shots(output_dir, output_dir / "shots.mp4")

                # ── DB push: shot_class + summary (after M4+M5) ───────────────
                if stdb is not None:
                    try:
                        hits_dicts = [asdict(e) for e in events]
                        shots_dicts = [asdict(e) for e in shot_events]
                        LOGGER.info("Pushing %d shots to SpacetimeDB (shot_class)", len(hits_dicts))
                        stdb.push_shot_classes(run_id, hits_dicts, shots_dicts)
                        LOGGER.info("Pushing summary to SpacetimeDB")
                        stdb.push_summary(run_id, frame_rows, hits_dicts, shots_dicts)
                    except Exception as exc:
                        LOGGER.warning("SpacetimeDB push (shots/summary) failed: %s", exc)

            if events:
                from module7_summary.rallies import build_rallies, write_rallies

                with partial.open(encoding="utf-8") as stream:
                    rally_frames = [json.loads(line) for line in stream if line.strip()]
                rallies, rally_info = build_rallies(
                    rally_frames, [asdict(e) for e in events],
                    [asdict(s) for s in shot_events], video_info["fps"],
                )
                write_rallies(output_dir, rallies)
                metadata["rallies"] = rally_info
                LOGGER.info("Rallies: %s", rally_info)
                if stdb is not None:
                    try:
                        stdb.push_rallies(run_id, rallies)
                    except Exception as exc:
                        LOGGER.warning("SpacetimeDB push (rallies) failed: %s", exc)

            partial.replace(output_dir / "frames.jsonl")
            if self.court is not None:
                metadata["court_model_inferences"] = getattr(self.court, "inference_count", 0)
            metadata.update(status="complete", processed_frames=count)

        except BaseException as exc:
            metadata.update(status="failed", processed_frames=count, error=str(exc))
            raise
        finally:
            write_json(metadata_path, metadata)

        return output_dir

    def _write_video(self, video, video_path, partial, calibration, samples, pose_threshold, court_threshold):
        trail = collections.deque(maxlen=_TRAIL_LEN)
        reference_index, reference = calibration if calibration is not None else (None, None)
        matrix = reference.homography if reference is not None else None
        count = 0
        writer = None
        try:
            with VideoReader(video) as reader, partial.open("w", encoding="utf-8") as stream:
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), reader.info.fps,
                    (reader.info.width, reader.info.height),
                )
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
                    stream.write(json.dumps(asdict(result), allow_nan=False) + "\n")
                    writer.write(render_frame(frame, result, trail, pose_threshold, court_threshold))
                    count += 1
                    if count % 100 == 0:
                        LOGGER.info("Processed %s frames", count)
        finally:
            if writer is not None:
                writer.release()
        if count == 0:
            raise ValueError("Video contains no decodable frames")
        LOGGER.info("Annotated video written to %s", video_path)
        return count

    @staticmethod
    def _write_shuttle_csv(path, samples, fps):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "frame_index", "timestamp_s", "x", "y", "visible", "confidence",
                "raw_x", "raw_y", "rectified", "source", "court_x_m", "court_y_m",
            ])
            for index, sample in sorted(samples.items()):
                writer.writerow([
                    index, index / fps,
                    *(sample.pixel or (None, None)),
                    int(sample.visible), sample.confidence,
                    *(sample.raw_pixel or (None, None)),
                    int(sample.rectified), sample.source,
                    *(sample.court_projection_m or (None, None)),
                ])
