"""Rally segmentation, rally outcomes and running score.

Built from M4 hits, M5 labels and court-projected shuttle positions.

An outcome is only claimed when the shuttle is seen coming to rest after the last
hit. Without that (clip ends mid-rally, shuttle lost, landing too close to a line)
the rally is "unknown" and no point is awarded: a wrong score is worse than a gap.

Limits worth knowing:
- Court projection is only meaningful once the shuttle is on the floor, which is
  why landings are read from a stationary shuttle, not a moving one.
- Lines are singles lines; this version supports singles matches only.
- Scores are per track ID. Track IDs follow position, so they can swap with the
  players when they change ends.
"""

import json
from math import hypot, isfinite

COURT_W = 6.10
COURT_L = 13.40
NET_Y = COURT_L / 2
SINGLES_INSET = 0.46

RALLY_GAP_S = 4.0         # A longer silence between hits starts a new rally.
MIN_HITS = 2              # A lone detection is usually a stray hit between points.
LANDING_WINDOW_S = 2.5    # How long after the last hit to look for a landing.
LANDING_DELAY_S = 0.2     # Ignore the racket contact itself.
STILL_S = 0.2             # The shuttle must stay put this long to count as landed.
STILL_RADIUS_M = 0.25
LINE_MARGIN_M = 0.10      # Closer than this to a line is too close to call.
NET_MARGIN_M = 0.15


def _shuttle_xy(frame):
    shuttle = frame.get("shuttle") or {}
    point = shuttle.get("court_projection_m")
    if not shuttle.get("visible") or not point or len(point) < 2:
        return None
    x, y = float(point[0]), float(point[1])
    return (x, y) if isfinite(x) and isfinite(y) else None


def _player_y(frame, track_id):
    for player in frame.get("players") or []:
        point = player.get("court_xy_m")
        if player.get("track_id") == track_id and point and len(point) >= 2:
            return float(point[1])
    return None


def find_landing(frames_by_index, start, stop, fps):
    """Return the (x, y) where the shuttle first rests in [start, stop], or None."""
    need = max(2, round(STILL_S * fps))
    run = []
    for index in range(start, stop + 1):
        frame = frames_by_index.get(index)
        point = _shuttle_xy(frame) if frame else None
        if point is None:
            run = []
            continue
        run = [p for p in run if hypot(p[0] - point[0], p[1] - point[1]) <= STILL_RADIUS_M] + [point]
        if len(run) >= need:
            return tuple(round(sum(c) / len(run), 3) for c in zip(*run))
    return None


def classify_landing(point, hitter_end):
    """Return 'winner', 'net', 'out' or 'unknown' for a landing and the hitter's end."""
    x, y = point
    far_hitter = hitter_end == "far"
    past_net = y > NET_Y + NET_MARGIN_M if far_hitter else y < NET_Y - NET_MARGIN_M
    if not past_net:
        short_of_net = y < NET_Y - NET_MARGIN_M if far_hitter else y > NET_Y + NET_MARGIN_M
        if abs(y - NET_Y) <= NET_MARGIN_M or short_of_net:
            return "net"
    left, right = SINGLES_INSET, COURT_W - SINGLES_INSET
    inside = (left + LINE_MARGIN_M <= x <= right - LINE_MARGIN_M
              and LINE_MARGIN_M <= y <= COURT_L - LINE_MARGIN_M)
    outside = (x < left - LINE_MARGIN_M or x > right + LINE_MARGIN_M
               or y < -LINE_MARGIN_M or y > COURT_L + LINE_MARGIN_M)
    return "winner" if inside else "out" if outside else "unknown"


def _hitter_end(hit, shot, frames_by_index):
    side = (shot or {}).get("player_side")
    if side in {"Top", "Bottom"}:
        return "far" if side == "Top" else "near"
    frame = frames_by_index.get(hit["frame_index"])
    y = _player_y(frame, hit.get("hitter_id")) if frame else None
    if y is None:
        return None
    return "far" if y < NET_Y else "near"


def _game_over(a, b):
    high, low = max(a, b), min(a, b)
    return high >= 30 or (high >= 21 and high - low >= 2)


def build_rallies(frames, hits, shots, fps):
    """Return (rallies, info). `frames`, `hits` and `shots` are lists of JSON-style dicts."""
    if not fps or fps <= 0:
        raise ValueError("fps must be positive")
    frames_by_index = {f["frame_index"]: f for f in frames}
    last_frame = max(frames_by_index, default=-1)
    shots_by_frame = {s["start_frame"]: s for s in shots}
    ordered = sorted(hits, key=lambda h: h["frame_index"])

    groups, current = [], []
    for hit in ordered:
        if current and hit["timestamp_s"] - current[-1]["timestamp_s"] > RALLY_GAP_S:
            groups.append(current)
            current = []
        current.append(hit)
    if current:
        groups.append(current)
    kept = [g for g in groups if len(g) >= MIN_HITS]

    rallies, game, points, reliable = [], 1, {1: 0, 2: 0}, True
    for index, group in enumerate(kept):
        last = group[-1]
        next_start = kept[index + 1][0]["frame_index"] if index + 1 < len(kept) else last_frame + 1
        start = last["frame_index"] + round(LANDING_DELAY_S * fps)
        stop = min(last["frame_index"] + round(LANDING_WINDOW_S * fps), next_start - 1, last_frame)
        landing = find_landing(frames_by_index, start, stop, fps)

        hitter = last.get("hitter_id") if last.get("hitter_id") in (1, 2) else None
        end = _hitter_end(last, shots_by_frame.get(last["frame_index"]), frames_by_index)
        outcome = classify_landing(landing, end) if landing and hitter and end else "unknown"
        winner = None
        if outcome == "winner":
            winner = hitter
        elif outcome in {"net", "out"}:
            winner = 3 - hitter

        before = dict(points)
        if winner is None:
            reliable = False
        else:
            points[winner] += 1
        rallies.append({
            "rally_index": index + 1,
            "game": game,
            "start_frame": group[0]["frame_index"],
            "end_frame": last["frame_index"],
            "start_s": group[0]["timestamp_s"],
            "end_s": last["timestamp_s"],
            "shot_count": len(group),
            "first_event_id": group[0]["event_id"],
            "last_event_id": last["event_id"],
            "last_hitter_id": hitter or 0,
            "winner_id": winner or 0,
            "outcome": outcome,
            "landing_xy_m": list(landing) if landing else None,
            "score_before": {"1": before[1], "2": before[2]},
            "score_after": {"1": points[1], "2": points[2]},
            "score_reliable": reliable,
        })
        # Only a fully counted game can be closed by the scoring rules.
        if reliable and _game_over(points[1], points[2]):
            game, points = game + 1, {1: 0, 2: 0}

    info = {
        "rallies": len(rallies),
        "decided": sum(1 for r in rallies if r["winner_id"]),
        "unknown": sum(1 for r in rallies if not r["winner_id"]),
        "dropped_short_segments": len(groups) - len(kept),
        "source": "landing_heuristic_v1",
    }
    return rallies, info


def write_rallies(output_dir, rallies):
    with (output_dir / "rallies.jsonl").open("w", encoding="utf-8") as stream:
        for rally in rallies:
            stream.write(json.dumps(rally) + "\n")


def publish_saved_run(run_dir, *, module="badminton", server="https://maincloud.spacetimedb.com", token=None):
    """Build rallies for a completed run directory, save them, and publish them."""
    from stdb_client import StdbClient

    def read_jsonl(name):
        path = run_dir / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    hits = read_jsonl("hits.jsonl")
    if not hits:
        raise ValueError(f"{run_dir} has no hits.jsonl; run hit detection first")
    rallies, info = build_rallies(read_jsonl("frames.jsonl"), hits, read_jsonl("shots.jsonl"), metadata["video"]["fps"])
    write_rallies(run_dir, rallies)
    client = StdbClient(module=module, server=server, token=token)
    client.clear_rallies(run_dir.name)
    client.push_rallies(run_dir.name, rallies)
    return info
