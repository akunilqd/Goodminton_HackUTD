"""BadmintonDB JnB_bone preprocessing and the application 18-to-6 taxonomy.

Uses BadmintonDB centered bbox poses, projected ankles, Top/Bottom slots,
masked bones and shared sampling/padding. Unlike upstream's in-court filter,
keep the two confirmed M1 identities when their feet are outside court bounds.
No keypoint-score cutoff: upstream uses the predicted coordinates directly.
"""

import numpy as np

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
BADDB_SEQ_LEN = 72
BONE_PAIRS: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
N_JOINTS = 17
N_BONES = len(BONE_PAIRS)
N_FEAT = (N_JOINTS + N_BONES) * 2

# Exact class_id order in BadmintonDB/after_generating/merged_strokes.csv.
_BASE_TYPES = [
    "Block", "Clear", "Drive", "Dropshot", "Net-Kill",
    "Net-Lift", "Net-Shot", "Serve", "Smash",
]
NATIVE_STROKE_TYPES = [
    f"{side}-{stroke}" for side in ("Bottom", "Top") for stroke in _BASE_TYPES
]
SIX_CLASS_GROUPS: dict[str, list[int]] = {
    "Smash": [4,  8, 13, 17],          # Net-Kill + Smash
    "Clear": [1,  5, 10, 14],          # Clear + Net-Lift (both lift to back court)
    "Drive": [2, 11],                  # Drive
    "Net":   [0,  3,  6, 9, 12, 15],  # Block + Dropshot + Net-Shot
    "Serve": [7, 16],                  # Serve
}
SIX_CLASSES = list(SIX_CLASS_GROUPS)


def group_probabilities(probs: np.ndarray) -> np.ndarray:
    """Marginalize sides/subtypes before argmax; no Unknown or renormalization."""
    probs = np.asarray(probs)
    if (probs.shape != (18,) or not np.isfinite(probs).all()
            or np.any(probs < 0) or not np.isclose(probs.sum(), 1.0, atol=1e-5)):
        raise ValueError("Expected 18 finite softmax probabilities summing to one")
    return np.array([probs[ids].sum() for ids in SIX_CLASS_GROUPS.values()])


def normalize_joints(keypoints: np.ndarray, bbox: np.ndarray, center_align=True) -> np.ndarray:
    """Single-player upstream normalization, including per-coordinate zero masks.

    Upstream masks BEFORE centering, so missing coordinates need not remain
    zero after centering. Preserve this for checkpoint compatibility.
    """
    dist = float(np.linalg.norm(bbox[2:] - bbox[:2]))
    if dist < 1e-6:
        return np.zeros((N_JOINTS, 2), dtype=np.float32)
    normalized = np.where(keypoints != 0.0, (keypoints - bbox[:2]) / dist, 0.0)
    if center_align:
        normalized -= ((bbox[:2] + bbox[2:]) / 2 - bbox[:2]) / dist
    return normalized.astype(np.float32)


def normalize_shuttlecock(xy: np.ndarray, v_width: int, v_height: int) -> np.ndarray:
    return np.asarray(xy, dtype=np.float32) / np.array([v_width, v_height], dtype=np.float32)


def normalize_position(court_xy: np.ndarray) -> np.ndarray:
    return np.asarray(court_xy) / np.array([COURT_WIDTH_M, COURT_LENGTH_M])


def compute_bones(joints_norm: np.ndarray) -> np.ndarray:
    bones = np.zeros((N_BONES, 2), dtype=np.float32)
    for i, (a, b) in enumerate(BONE_PAIRS):
        start, end = joints_norm[a], joints_norm[b]
        bones[i] = np.where((start != 0.0) & (end != 0.0), end - start, 0.0)
    return bones


def ordered_players(row: dict) -> list[tuple]:
    """Return (player, joints, bbox, normalized position), sorted Top then Bottom.

    M3 H already maps source pixels to metres. Project EACH ankle before taking
    their mean: perspective projection and averaging do not commute.
    M1 assigns persistent IDs 1 and 2 to the selected match players. Require
    exactly one usable detection per ID; never substitute an untracked person.
    Keep finite outside-court positions without clipping or inventing poses.
    """
    matrix = np.asarray((row.get("court") or {}).get("homography"), dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or np.linalg.matrix_rank(matrix) != 3):
        return []
    candidates = []
    tracked = [p for p in row.get("players") or [] if p.get("track_id") in (1, 2)]
    if len(tracked) != 2 or {p["track_id"] for p in tracked} != {1, 2}:
        return []
    for player in tracked:
        kp = np.asarray(player.get("keypoints"), dtype=float)
        bbox = np.asarray(player.get("bbox_xyxy"), dtype=float)
        if (kp.ndim != 2 or kp.shape[0] != N_JOINTS or kp.shape[1] < 2
                or bbox.shape != (4,) or not np.isfinite(kp[:, :2]).all()
                or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2])
                or np.any(np.all(kp[-2:, :2] == 0, axis=1))):
            continue
        ankles = np.column_stack((kp[-2:, :2], np.ones(2))) @ matrix.T
        if np.any(np.abs(ankles[:, 2]) < 1e-8):
            continue
        position = normalize_position((ankles[:, :2] / ankles[:, 2:]).mean(axis=0))
        if np.isfinite(position).all():
            candidates.append((player, kp[:, :2], bbox, position))
    if len(candidates) != 2:
        return []
    return sorted(candidates, key=lambda item: item[3][1])


def build_input_arrays(
    frame_rows: list[dict], seq_len: int, v_width: int, v_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (T,2,36,2), (T,2), (T,2,2) tensors and unpadded sequence length.

    Caller retains missing frame indices as empty rows. Sample all modalities
    with upstream stride, then zero-pad; video_len excludes padding only.
    """
    if seq_len < 1 or v_width <= 0 or v_height <= 0:
        raise ValueError("Sequence length and video dimensions must be positive")
    length = len(frame_rows)
    stride = 1
    if length > seq_len:
        stride = length // seq_len + int(length % seq_len > seq_len // 2)
    rows = frame_rows[::stride][:seq_len]
    pose = np.zeros((seq_len, 2, N_JOINTS + N_BONES, 2), dtype=np.float32)
    shuttle = np.zeros((seq_len, 2), dtype=np.float32)
    pos = np.zeros((seq_len, 2, 2), dtype=np.float32)
    for t, row in enumerate(rows):
        players = ordered_players(row)
        if not players:
            # Upstream also zeros shuttle when player detection fails.
            continue
        for slot, (_, xy, bbox, position) in enumerate(players):
            joints = normalize_joints(xy, bbox)
            pose[t, slot] = np.concatenate((joints, compute_bones(joints)), axis=0)
            pos[t, slot] = position
        sample = row.get("shuttle") or {}
        pixel = np.asarray(sample.get("pixel"), dtype=float)
        if sample.get("visible") and pixel.shape == (2,) and np.isfinite(pixel).all():
            shuttle[t] = normalize_shuttlecock(pixel, v_width, v_height)
    return pose, shuttle, pos, len(rows)
