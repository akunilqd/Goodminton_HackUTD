"""CPU-only heuristic hit detection. No neural models or video decoding required.

Windowed velocity fits preserve the contact discontinuity. Missing frames remain
on the timeline; only short, bounded gaps are interpolated. Rally boundaries are
explicit annotations, never inferred just from tracking loss. Input is JSON-like
FrameResult records, so saved runs and the video pipeline share exactly one path.
"""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from schemas import HitEvent


class HitDetector:
    def __init__(self, settings):
        self.settings = settings

    def detect(self, rows, fps, width, height, rallies=None):
        started = time.perf_counter()
        if not all(math.isfinite(v) and v > 0 for v in (fps, width, height)):
            raise ValueError("M4 requires positive finite FPS and image dimensions")
        s = self.settings
        points, wrists, sources = [], [], []
        for i, row in enumerate(rows):
            if row["frame_index"] != i or not math.isclose(
                row["timestamp_s"], i / fps, abs_tol=1e-5
            ):
                raise ValueError(f"M4 requires contiguous zero-based CFR frames; bad frame {i}")
            if row.get("players") is None or row.get("shuttle") is None:
                raise ValueError("M4 requires enabled M1 and M2 outputs (empty detections are allowed)")
            shuttle = row["shuttle"]
            point = shuttle.get("pixel") if shuttle.get("visible") else None
            if point is not None and (len(point) != 2 or not np.isfinite(point).all()):
                raise ValueError(f"Invalid shuttle point at frame {i}")
            if point is not None and not (0 <= point[0] < width and 0 <= point[1] < height):
                point = None
            points.append(point if point is not None else (np.nan, np.nan))
            sources.append(shuttle.get("source", "missing"))
            frame_wrists = []
            for player in row["players"]:
                box = np.asarray(player["bbox_xyxy"], dtype=float)
                joints = np.asarray(player["keypoints"], dtype=float)
                if box.shape != (4,) or joints.shape != (17, 3) or not np.isfinite(box).all():
                    raise ValueError(f"Invalid M1 pose/box at frame {i}")
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                scale = float(np.linalg.norm(box[2:] - box[:2]))
                for joint in joints[[9, 10]]:
                    if (np.isfinite(joint).all() and joint[2] >= s.pose_confidence
                            and 0 <= joint[0] < width and 0 <= joint[1] < height
                            and np.any(joint[:2] != 0)):
                        frame_wrists.append((player["track_id"], joint[:2], scale))
            wrists.append(frame_wrists)
        n = len(points)
        if n == 0:
            raise ValueError("M4 input contains no frames")
        xy = np.asarray(points, dtype=float)
        original_valid = np.isfinite(xy).all(axis=1)
        rally_ids = [None] * n
        groups = np.zeros(n, dtype=int)
        if rallies is not None:
            if not isinstance(rallies, list):
                raise ValueError("Rally annotations must be a JSON list")
            groups.fill(-1)
            last_end = -1
            seen = set()
            for k, rally in enumerate(rallies):
                if not isinstance(rally, dict) or not {"start_frame", "end_frame", "rally_id"}.issubset(rally):
                    raise ValueError("Each rally needs rally_id, start_frame and end_frame")
                start, end, rid = rally["start_frame"], rally["end_frame"], rally["rally_id"]
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start <= end < n or start <= last_end
                        or not isinstance(rid, (str, int)) or isinstance(rid, bool) or rid in seen):
                    raise ValueError("Rallies need unique IDs and sorted, non-overlapping inclusive frame ranges")
                groups[start:end + 1] = k
                rally_ids[start:end + 1] = [rid] * (end - start + 1)
                seen.add(rid)
                last_end = end
        max_gap = int(math.floor(s.hit_max_gap_s * fps + 1e-9))
        interpolated = np.zeros(n, dtype=bool)
        i = 0
        while i < n:
            if original_valid[i]:
                i += 1
                continue
            start = i
            while i < n and not original_valid[i]:
                i += 1
            if (0 < start and i < n and i - start <= max_gap
                    and groups[start - 1] >= 0
                    and np.all(groups[start - 1:i + 1] == groups[start - 1])):
                for j in range(start, i):
                    xy[j] = xy[start - 1] + (xy[i] - xy[start - 1]) * (j - start + 1) / (i - start + 1)
                interpolated[start:i] = True
        valid = np.isfinite(xy).all(axis=1) & (groups >= 0)
        window = max(2, round(s.hit_velocity_window_s * fps))
        wrist_window = max(0, round(s.hit_wrist_window_s * fps))
        merge = max(1, round(s.hit_merge_window_s * fps))
        normalized = xy / math.hypot(width, height)
        t = np.arange(window + 1, dtype=float) / fps
        t -= t.mean()
        denominator = float(t @ t)
        before = np.full_like(xy, np.nan)
        after = np.full_like(xy, np.nan)
        eligible = []
        for i in range(window, n - window):
            a, b = i - window, i + window + 1
            if valid[a:b].all() and np.all(groups[a:b] == groups[i]):
                before[i] = t @ normalized[a:i + 1] / denominator
                after[i] = t @ normalized[i:b] / denominator
                eligible.append(i)
        delta = np.linalg.norm(after - before, axis=1)
        finite_delta = delta[eligible]
        if len(finite_delta):
            median = float(np.median(finite_delta))
            mad = float(np.median(np.abs(finite_delta - median)))
            change_threshold = max(s.hit_min_velocity_change, median + 2.5 * 1.4826 * mad)
        else:
            change_threshold = s.hit_min_velocity_change

        candidates = []
        for i in eligible:
            speed_before, speed_after = np.linalg.norm(before[i]), np.linalg.norm(after[i])
            angle = 0.
            if min(speed_before, speed_after) >= s.hit_min_speed:
                cosine = np.dot(before[i], after[i]) / (speed_before * speed_after)
                angle = math.degrees(math.acos(float(np.clip(cosine, -1, 1))))
            serve = speed_before < s.hit_min_speed and speed_after >= 2 * s.hit_min_speed
            change = delta[i] >= change_threshold
            turn = angle >= s.hit_turn_degrees
            if not (change or turn or serve):
                continue
            if max(speed_before, speed_after) < s.hit_min_speed:
                continue
            distances = {}
            for j in range(max(0, i - wrist_window), min(n, i + wrist_window + 1)):
                if not valid[j] or groups[j] != groups[i]:
                    continue
                for identity, wrist, scale in wrists[j]:
                    distance = float(np.linalg.norm(xy[j] - wrist) / scale)
                    distances[identity] = min(distances.get(identity, float("inf")), distance)
            ranked = sorted(distances.items(), key=lambda item: item[1])
            near = bool(ranked) and ranked[0][1] <= s.hit_wrist_radius
            # Known wrists far away reject floor/net bounces. Without wrists only
            # retain strong trajectory evidence, with unknown hitter explicitly.
            if ranked and not near:
                continue
            if not near and (serve or not (delta[i] >= 2 * change_threshold and turn)):
                continue
            proximity = max(0., 1 - ranked[0][1] / s.hit_wrist_radius) if near else 0.
            # Keep change magnitude unsaturated so a broad turning window does
            # not tie on every frame and bias NMS toward an early contact time.
            motion = .7 * delta[i] / (delta[i] + change_threshold) + .3 * angle / 180.
            if serve:
                motion = max(motion, .7)
            a, b = i - window, i + window + 1
            repaired_fraction = sum(src == "inpaintnet" for src in sources[a:b]) / (b - a)
            quality = 1. - .25 * repaired_fraction - .5 * float(interpolated[a:b].mean())
            score = (.5 * motion + .35 * proximity + .15 * quality if near
                     else .85 * motion + .15 * quality)
            if score < s.hit_min_score:
                continue
            hitter, hitter_score, flags = None, 0., []
            if near:
                separation = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else float("inf")
                if separation >= s.hit_hitter_margin and ranked[0][0] is not None:
                    hitter = ranked[0][0]
                    hitter_score = proximity
                else:
                    flags.append("ambiguous_hitter")
            else:
                flags.append("missing_wrist_evidence")
            if repaired_fraction:
                flags.append("repaired_shuttle_context")
            if interpolated[a:b].any():
                flags.append("interpolated_shuttle_context")
            if serve:
                flags.append("possible_serve")
            if rally_ids[i] is None:
                flags.append("rally_unverified")
            candidates.append(HitEvent(0, rally_ids[i], i, i / fps, hitter,
                                       float(score), float(hitter_score), window, flags))

        # Greedy temporal NMS: bounded radius, not transitive clustering that
        # could merge an arbitrarily long chain of distinct rapid exchanges.
        suppressed = np.zeros(n, dtype=bool)
        kept = []
        for event in sorted(candidates, key=lambda e: (-e.hit_score, -delta[e.frame_index], e.frame_index)):
            i = event.frame_index
            if suppressed[i]:
                continue
            kept.append(event)
            a, b = max(0, i - merge), min(n, i + merge + 1)
            suppressed[a:b] |= groups[a:b] == groups[i]
        kept.sort(key=lambda e: e.frame_index)
        for k, event in enumerate(kept):
            event.event_id = k + 1
            if k and event.hitter_id is not None and event.hitter_id == kept[k - 1].hitter_id:
                previous = kept[k - 1]
                if (event.rally_id == previous.rally_id
                        and valid[previous.frame_index:event.frame_index + 1].all()):
                    event.quality_flags.append("repeated_hitter_review")
        gaps = []
        i = 0
        while i < n:
            if valid[i]:
                i += 1
                continue
            start = i
            while i < n and not valid[i]:
                i += 1
            gaps.append({"start_frame": start, "end_frame": i - 1})
        summary = {
            "algorithm": "heuristic_v1", "frame_count": n, "hit_count": len(kept),
            "candidate_count": len(candidates), "unknown_hitter_count": sum(e.hitter_id is None for e in kept),
            "interpolated_frames": int(interpolated.sum()), "unusable_ranges": gaps,
            "evaluated_frames": len(eligible), "unevaluated_frames": n - len(eligible),
            "boundary_context_frames": window, "velocity_change_threshold": change_threshold,
            "boundary_policy": "No event without full before/after context; inspect unevaluated boundary frames",
            "camera_cut_detection": False,
            "rallies_verified": rallies is not None, "scores_are_probabilities": False,
            "processing_seconds": time.perf_counter() - started,
            "settings": {k: v for k, v in asdict(s).items() if k.startswith("hit_") or k == "pose_confidence"},
        }
        return kept, summary


def write_hits(output, events, summary):
    """Caller owns a fresh output directory; publish hits only after writing."""
    output = Path(output)
    partial = output / "hits.partial.jsonl"
    with partial.open("x", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(asdict(event), allow_nan=False) + "\n")
    partial.replace(output / "hits.jsonl")
    (output / "hit_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def detect_saved_run(settings, run_dir, output_dir, rallies_path=None):
    """Read-only source replay: does not load M1-M3 models or modify source run."""
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Output directory already exists; select a new directory")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("M4 requires a complete source run")
    if not all(metadata.get("modules", {}).get(m) for m in ("m1", "m2")):
        raise ValueError("Source run must contain M1 and M2; homography is optional for M4")
    video = metadata["video"]
    rallies = None if rallies_path is None else json.loads(Path(rallies_path).read_text(encoding="utf-8"))
    if rallies is not None and not isinstance(rallies, list):
        raise ValueError("Rally annotations must be a JSON list")
    with (run_dir / "frames.jsonl").open(encoding="utf-8") as stream:
        events, summary = HitDetector(settings).detect(
            (json.loads(line) for line in stream), video["fps"], video["width"], video["height"], rallies
        )
    if summary["frame_count"] != metadata["processed_frames"]:
        raise ValueError("Source metadata frame count does not match frames.jsonl")
    summary["source_run"] = str(run_dir)
    summary["video"] = video
    output_dir.mkdir(parents=True, exist_ok=False)
    write_hits(output_dir, events, summary)
    return summary
