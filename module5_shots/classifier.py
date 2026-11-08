"""M5: BST shot classification with context before and after each target hit.

Requires the BST repo cloned at settings.bst_repo:
  git clone https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer <bst_repo>

BadmintonDB weights (18 native classes, grouped into 6 application classes):
  https://drive.google.com/file/d/1KXY62Q-5kSkgqUiJY88DApdsI2qDqZC5/view
Save as weights/bst_CG_AP_JnB_bone.pt.
"""

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from config import Settings
from schemas import ShotEvent
from .preprocessing import (
    BADDB_SEQ_LEN,
    NATIVE_STROKE_TYPES,
    N_BONES,
    N_JOINTS,
    SIX_CLASS_GROUPS,
    SIX_CLASSES,
    build_input_arrays,
    group_probabilities,
    ordered_players,
)

LOGGER = logging.getLogger(__name__)

_IN_DIM = (N_JOINTS + N_BONES) * 2   # 72, JnB_bone mode
_N_CLASSES = 18


def input_window(
    hits: list[dict], index: int, fps: float, first_frame: int, last_frame: int,
    settings: Settings,
) -> tuple[int, int, list[str]]:
    """Inclusive model context; never change the target hit's display interval.

Adapt ShuttleSet's previous-hit -> next-hit + extension strategy for this
BadmintonDB experiment. This is not a claim of matching BadmintonDB annotations.
Respect available frames and known neighboring rally IDs. Without rally IDs,
boundaries remain unverified, as in M4's output.
"""
    target = hits[index]["frame_index"]
    rally_id = hits[index].get("rally_id")
    limit = max(1, int(fps * settings.bst_context_limit_s))
    fallback = max(1, int(fps * settings.bst_context_fallback_s))
    extension = max(1, int(fps * settings.bst_context_after_next_s))
    previous = hits[index - 1] if index else None
    following = hits[index + 1] if index + 1 < len(hits) else None
    has_previous = previous is not None and previous.get("rally_id") == rally_id
    has_following = following is not None and following.get("rally_id") == rally_id
    left = previous["frame_index"] if has_previous else target - fallback
    # The upstream clip end is exclusive; our serialized bounds are inclusive.
    right = following["frame_index"] + extension - 1 if has_following else target + fallback - 1
    left = max(left, target - limit)
    right = min(right, target + limit + extension - 1)
    lower, upper = first_frame, last_frame
    for j in range(index - 1, -1, -1):
        neighbor = hits[j]
        if neighbor["frame_index"] < left:
            break
        if neighbor.get("rally_id") != rally_id:
            lower = max(lower, neighbor["frame_index"] + 1)
            break
    for j in range(index + 1, len(hits)):
        neighbor = hits[j]
        if neighbor["frame_index"] > right:
            break
        if neighbor.get("rally_id") != rally_id:
            upper = min(upper, neighbor["frame_index"] - 1)
            break
    flags = []
    if not has_previous:
        flags.append("no_previous_hit_in_rally")
    if not has_following:
        flags.append("no_next_hit_in_rally")
    if rally_id is None:
        flags.append("rally_boundaries_unverified")
    if left < lower or right > upper:
        flags.append("input_context_clipped")
    return max(left, lower), min(right, upper), flags


def _import_bst(repo_path: Path, model_name: str):
    stroke_cls = repo_path / "stroke_classification"
    if not stroke_cls.is_dir():
        raise RuntimeError(
            f"BST repo not found at {repo_path}.\n"
            "Clone with:\n"
            f"  git clone https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer {repo_path}"
        )
    if str(stroke_cls) not in sys.path:
        sys.path.insert(0, str(stroke_cls))
    try:
        from model.bst import BST, BST_AP, BST_CG, BST_CG_AP  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"Cannot import BST model from {stroke_cls}: {exc}") from exc
    choices = {"BST": BST, "BST_CG": BST_CG, "BST_AP": BST_AP, "BST_CG_AP": BST_CG_AP}
    if model_name not in choices:
        raise ValueError(f"bst_model must be one of {list(choices)}; got {model_name!r}")
    return choices[model_name]


class ShotClassifier:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.seq_len = settings.bst_seq_len
        if self.seq_len != BADDB_SEQ_LEN:
            raise ValueError("BadmintonDB checkpoint requires bst_seq_len=72")
        self.last_run_summary = {}
        dev = torch.device(settings.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("BST: CUDA unavailable; explicitly set device='cpu' to use CPU")
        if dev.type not in {"cpu", "cuda"}:
            raise ValueError("BST device must be cpu or cuda:N")
        self.device = str(dev)

        bst_repo = Path(settings.bst_repo)
        bst_weights = Path(settings.bst_weights)
        model_name = getattr(settings, "bst_model", "BST_CG_AP")

        ModelClass = _import_bst(bst_repo, model_name)
        self.net = ModelClass(
            in_dim=_IN_DIM,
            n_class=_N_CLASSES,
            seq_len=self.seq_len,
            depth_tem=2,
            depth_inter=1,
        ).to(self.device)

        if not bst_weights.is_file():
            raise ValueError(
                f"BST weights not found at {bst_weights}.\n"
                "Download from: https://drive.google.com/file/d/"
                "1KXY62Q-5kSkgqUiJY88DApdsI2qDqZC5/view"
            )
        state = torch.load(str(bst_weights), map_location=self.device, weights_only=True)
        try:
            self.net.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint does not match BadmintonDB (18 classes, 72 frames, "
                f"{model_name}). Use the matching BadmintonDB weight."
            ) from exc
        self.net.eval()
        LOGGER.info("BST %s loaded from %s (device=%s)", model_name, bst_weights, self.device)

    def _predict(self, prepared: list[tuple]) -> list[np.ndarray]:
        """Return 18-class softmax for each window; each sample keeps its own padding mask."""
        probabilities = []
        for offset in range(0, len(prepared), self.settings.bst_batch_size):
            chunk = prepared[offset:offset + self.settings.bst_batch_size]
            pose, shuttle, pos, lengths = zip(*(window[7:10] + window[11:] for window in chunk))
            hp = torch.from_numpy(np.stack(pose)).to(self.device)
            hp = hp.reshape(len(chunk), self.seq_len, 2, _IN_DIM)
            sh = torch.from_numpy(np.stack(shuttle)).to(self.device)
            po = torch.from_numpy(np.stack(pos)).to(self.device)
            vl = torch.tensor(lengths, dtype=torch.long, device=self.device)
            logits = self.net(hp, sh, po, vl)
            if logits.shape != (len(chunk), _N_CLASSES) or not torch.isfinite(logits).all():
                raise RuntimeError("BST must return 18 finite logits per shot")
            probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return probabilities

    @torch.inference_mode()
    def classify(
        self,
        hits: list[dict],
        frames: dict[int, dict],
        fps: float,
        v_width: int,
        v_height: int,
    ) -> list[ShotEvent]:
        """
        Classify each hit from surrounding context; display labels between hits.

        hits:   sorted list of HitEvent dicts (from hits.jsonl)
        frames: {frame_index: frame_dict} from frames.jsonl
        """
        if not np.isfinite(fps) or fps <= 0 or v_width <= 0 or v_height <= 0:
            raise ValueError("Video fps and dimensions must be positive")
        started = time.perf_counter()
        hits = sorted(hits, key=lambda h: h["frame_index"])
        if len({h["frame_index"] for h in hits}) != len(hits):
            raise ValueError("Hit frame indices must be unique")
        shots: list[ShotEvent] = []
        skipped = []
        first_frame = max(0, min(frames, default=0))
        last_frame = max(frames, default=-1)
        min_seg = self.settings.bst_min_seg_frames
        conf_thr = self.settings.bst_conf_threshold
        prepared = []

        for i, hit in enumerate(hits):
            start = hit["frame_index"]
            same_rally = (i + 1 < len(hits)
                          and hits[i + 1].get("rally_id") == hit.get("rally_id"))
            boundary = hits[i + 1]["frame_index"] - 1 if i + 1 < len(hits) else last_frame
            end = min(boundary, last_frame)
            if not same_rally:
                end = min(end, start + self.seq_len - 1)
            reason = None
            if start < first_frame or end < start:
                reason = "hit_outside_frames"
            if reason:
                skipped.append({"shot_id": i + 1, "start_frame": start, "reason": reason})
                continue
            input_start, input_end, flags = input_window(
                hits, i, fps, first_frame, last_frame, self.settings
            )
            # Keep input context independent of non-overlapping display intervals.
            # Empty rows retain missing timestamps; never compress the timeline.
            frame_rows = [frames.get(f, {}) for f in range(input_start, input_end + 1)]
            pose, shuttle, pos, video_len = build_input_arrays(
                frame_rows, self.seq_len, v_width, v_height
            )
            valid = np.any(pos != 0, axis=(1, 2))
            if not valid.any():
                skipped.append({"shot_id": i + 1, "start_frame": start,
                                "reason": "no_valid_player_pair_or_court"})
                continue
            prepared.append((i, hit, end, input_start, input_end, flags, frame_rows,
                             pose, shuttle, pos, valid, video_len))

        probabilities = self._predict(prepared)
        for window, probs in zip(prepared, probabilities):
            i, hit, end, input_start, input_end, flags, frame_rows, _, _, pos, valid, video_len = window
            start = hit["frame_index"]
            grouped = group_probabilities(probs)
            class_id = int(np.argmax(grouped))
            confidence = float(grouped[class_id])
            raw_id = int(np.argmax(probs))
            # Hitter identity belongs to M4. Do not infer it from native class side.
            side = None
            if hit.get("hitter_id") is not None:
                for slot, (player, *_) in enumerate(ordered_players(frames.get(start, {}))):
                    if player.get("track_id") == hit["hitter_id"]:
                        side = ("Top", "Bottom")[slot]
            if input_end - input_start + 1 < min_seg:
                flags.append("short_segment")
            if np.any(valid & np.any((pos <= -0.01) | (pos >= 1.01), axis=(1, 2))):
                flags.append("outside_court_player_positions")
            if any(not row for row in frame_rows):
                flags.append("missing_frames_zero_filled")
            if not valid[:video_len].all():
                flags.append("unusable_player_frames_zero_filled")
            shot = ShotEvent(
                shot_id=i + 1, start_frame=start, end_frame=end,
                hitter_id=hit.get("hitter_id"), player_side=side,
                class_id=class_id, shot_type_raw=NATIVE_STROKE_TYPES[raw_id],
                shot_type_en=SIX_CLASSES[class_id], confidence=confidence,
                raw_class_id=raw_id, raw_confidence=float(probs[raw_id]),
                class_probabilities={name: float(p) for name, p in zip(SIX_CLASSES, grouped)},
                low_confidence=confidence < conf_thr, quality_flags=flags,
                input_start_frame=input_start, input_end_frame=input_end,
                input_sample_count=video_len,
            )
            shots.append(shot)
            LOGGER.info("Shot %d hit=%d input=[%d-%d]: %s (conf=%.3f%s)",
                        shot.shot_id, start, input_start, input_end, shot.shot_type_en, confidence,
                        ", low" if shot.low_confidence else "")

        self.last_run_summary = {
            "input_hit_count": len(hits), "skipped_count": len(skipped),
            "skipped_segments": skipped,
            "processing_seconds": time.perf_counter() - started,
            "device": self.device,
            "segment_policy": "hit_to_before_next_hit; rally_tail_max_72_frames; clamp_to_video",
            "input_window_policy": "previous_hit_to_next_hit_plus_context_v1",
            "input_window_matches_baddb_annotations": False,
        }
        LOGGER.info("M5: %d/%d shots classified, %d skipped (see shot_summary.json)",
                    len(shots), len(hits), len(skipped))
        return shots


def write_shots(
    output_dir: Path, shot_events: list[ShotEvent], settings: Settings,
    diagnostics: dict | None = None,
) -> dict:
    """Write predictions and their taxonomy/settings; preserve full probabilities."""
    output_dir = Path(output_dir)
    partial = output_dir / "shots.partial.jsonl"
    with partial.open("x", encoding="utf-8") as stream:
        for event in shot_events:
            stream.write(json.dumps(asdict(event), allow_nan=False) + "\n")
    partial.replace(output_dir / "shots.jsonl")
    summary = {
        "shot_count": len(shot_events), "model": settings.bst_model,
        "weights": str(settings.bst_weights), "seq_len": settings.bst_seq_len,
        "n_classes": len(SIX_CLASSES), "native_n_classes": _N_CLASSES,
        "class_scheme": "BadmintonDB_18_to_6_v1", "class_names": SIX_CLASSES,
        "native_class_names": NATIVE_STROKE_TYPES, "class_groups": SIX_CLASS_GROUPS,
        "confidence_threshold": settings.bst_conf_threshold,
        "threshold_policy": "flag_only_keep_argmax",
        "confidence_is_calibrated": False,
        "low_confidence_count": sum(s.low_confidence for s in shot_events),
        "short_segment_threshold_frames": settings.bst_min_seg_frames,
        "short_segment_basis": "input_window_before_padding",
        "input_context_seconds": {
            "limit_per_side": settings.bst_context_limit_s,
            "fallback_without_neighbor": settings.bst_context_fallback_s,
            "extension_after_next_hit": settings.bst_context_after_next_s,
        },
        "short_segment_count": sum("short_segment" in s.quality_flags for s in shot_events),
        "outside_court_segment_count": sum(
            "outside_court_player_positions" in s.quality_flags for s in shot_events
        ),
        "short_segment_policy": "infer_with_padding_and_flag",
        "preprocessing_policy": "baddb_tracked_pair_relaxed_court_v2",
        **(diagnostics or {}), "shots": [asdict(s) for s in shot_events],
    }
    summary_partial = output_dir / "shot_summary.partial.json"
    with summary_partial.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    summary_partial.replace(output_dir / "shot_summary.json")
    return summary


def classify_saved_run(settings: Settings, run_dir: Path, output_dir: Path) -> dict:
    """Run M5 on a completed run directory that contains frames.jsonl and hits.jsonl."""
    run_dir    = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if output_dir.exists():
        raise ValueError("Output directory already exists; select a new directory")

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("M5 requires a complete source run")
    if not all(metadata.get("modules", {}).get(m) for m in ("m1", "m2", "m3", "m4")):
        raise ValueError("Source run must contain M1, M2, M3 and M4 outputs")

    video_info = metadata["video"]
    fps, v_width, v_height = video_info["fps"], video_info["width"], video_info["height"]

    frames: dict[int, dict] = {}
    with (run_dir / "frames.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            frames[r["frame_index"]] = r

    hits: list[dict] = []
    with (run_dir / "hits.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                hits.append(json.loads(line))
    hits.sort(key=lambda h: h["frame_index"])

    classifier = ShotClassifier(settings)
    shot_events = classifier.classify(hits, frames, fps, v_width, v_height)

    output_dir.mkdir(parents=True, exist_ok=False)
    shot_summary = write_shots(output_dir, shot_events, settings, classifier.last_run_summary)

    summary = {
        "source_run": str(run_dir),
        "status": "complete",
        "modules": {"m5": True},
        "shot_classification": {k: v for k, v in shot_summary.items() if k != "shots"},
        "shot_count": len(shot_events),
        "video": video_info,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary
