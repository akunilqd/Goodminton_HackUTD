"""Per-frame player and shuttle metrics drawn over the match video."""

import bisect
from math import hypot

import cv2
import numpy as np

from module6_analytics.speed import _frame_from_dict, _shuttle_speed_m_s, player_speed_m_s

PLAYER_IDS = (1, 2)
KEYS = ("speed", "accel", "dist")
HISTORY_S = 3.0
WHITE = (255, 255, 255)
MUTED = (190, 190, 190)
PANEL = (20, 20, 20)
GRID = (60, 60, 60)
P_COLORS = {1: (255, 180, 60), 2: (80, 200, 255)}
SHUTTLE_COLOR = (0, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def compute_metrics(frame_rows: list[dict], fps: float) -> dict[int, dict]:
    frames = sorted((_frame_from_dict(r) for r in frame_rows), key=lambda f: f.frame_index)

    speed: dict[int, dict] = {}
    for prev, cur, nxt in zip(frames, frames[1:], frames[2:]):
        values = {tid: player_speed_m_s(prev, cur, nxt, tid) for tid in PLAYER_IDS}
        values["shuttle"] = _shuttle_speed_m_s(prev, cur, nxt)
        speed[cur.frame_index] = values

    metrics: dict[int, dict] = {}
    last_xy: dict[int, tuple[float, float]] = {}
    distance = {tid: 0.0 for tid in PLAYER_IDS}
    for fr in frames:
        current = speed.get(fr.frame_index, {})
        previous = speed.get(fr.frame_index - 1, {})
        row: dict = {}
        for tid in PLAYER_IDS:
            xy = next((p.court_xy_m for p in fr.players or []
                       if p.track_id == tid and p.court_xy_m), None)
            if xy is not None:
                if tid in last_xy:
                    distance[tid] += hypot(xy[0] - last_xy[tid][0], xy[1] - last_xy[tid][1])
                last_xy[tid] = xy
            v, pv = current.get(tid), previous.get(tid)
            accel = (v - pv) * fps if v is not None and pv is not None else None
            row[tid] = {"speed": v, "accel": accel, "dist": distance[tid]}
        row["shuttle"] = current.get("shuttle")
        metrics[fr.frame_index] = row
    return metrics


def scales_from(metrics: dict[int, dict]) -> dict[str, float]:
    player = [r[tid]["speed"] for r in metrics.values() for tid in PLAYER_IDS
              if r[tid]["speed"] is not None]
    shuttle = [r["shuttle"] for r in metrics.values() if r["shuttle"] is not None]
    return {
        "player": max(player, default=1.0) or 1.0,
        "shuttle": max(shuttle, default=1.0) or 1.0,
    }


def shot_at(shots: list[dict], starts: list[int], index: int) -> dict | None:
    k = bisect.bisect_right(starts, index) - 1
    return shots[k] if k >= 0 else None


def draw_metrics(frame, metrics, index, shot, names, fps, scales):
    row = metrics.get(index) or metrics.get(index - 1)
    window = _window(metrics, index, fps)
    height = frame.shape[0]
    s = max(0.5, frame.shape[1] / 1920)
    pad = round(14 * s)
    width_px = round(360 * s)
    gap = round(10 * s)

    shuttle_h = round(104 * s)
    shuttle_y = height - pad - shuttle_h
    shot_h = round(220 * s)
    shot_y = shuttle_y - gap - shot_h
    players_h = round(252 * s)
    players_y = shot_y - gap - players_h

    _draw_players(frame, row, window, names, scales, s, pad, players_y, width_px, players_h)
    _draw_shot(frame, shot, names, fps, s, pad, shot_y, width_px, shot_h)
    _draw_shuttle(frame, row, window, scales, s, pad, shuttle_y, width_px, shuttle_h)


def _window(metrics, index, fps):
    k = max(2, round(fps * HISTORY_S))
    return [metrics.get(i) for i in range(index - k + 1, index + 1)]


def _draw_players(frame, row, window, names, scales, s, x, y, w, h):
    _panel(frame, x, y, w, h)
    _text_center(frame, "Players Metrics", x + w // 2, y + round(32 * s), 0.7 * s, WHITE, 2)

    columns = [(1, x + round(118 * s)), (2, x + round(236 * s))]
    for tid, cx in columns:
        _text(frame, names.get(tid, f"P{tid}"), cx, y + round(68 * s), 0.6 * s, WHITE, 2)
    for k, label in enumerate(("Speed", "Accel.", "Distance")):
        ly = y + round((104 + 30 * k) * s)
        _text(frame, label, x + round(14 * s), ly, 0.55 * s, MUTED)
        for tid, cx in columns:
            _text(frame, _value(row, tid, KEYS[k]), cx, ly, 0.55 * s, WHITE)

    spark_y, spark_h = y + round(200 * s), round(40 * s)
    spark_w = round(160 * s)
    for tid, sx in ((1, x + round(14 * s)), (2, x + round(186 * s))):
        values = [r[tid]["speed"] if r else None for r in window]
        _sparkline(frame, values, sx, spark_y, spark_w, spark_h,
                   P_COLORS[tid], scales["player"], s)


def _draw_shot(frame, shot, names, fps, s, x, y, w, h):
    _panel(frame, x, y, w, h)
    _text(frame, "Shot Analysis", x + round(14 * s), y + round(34 * s), 0.7 * s, WHITE, 2)
    if shot is None:
        _text(frame, "No shot yet", x + round(20 * s), y + round(80 * s), 0.65 * s, MUTED)
        return

    pid = shot.get("hitter_id")
    player = f"{names.get(pid, f'P{pid}')} (ID: {pid})" if pid else "P?"
    fields = [
        ("Player", player),
        ("Shot Type", f"{shot['shot_type_en']} shot"),
        ("Frame", str(shot["start_frame"])),
        ("Time", _mmss(shot["start_frame"] / fps) if fps else "--"),
        ("Confidence", f"{shot['confidence'] * 100:.0f}%"),
    ]
    for k, (label, val) in enumerate(fields):
        ly = y + round((76 + 32 * k) * s)
        _text(frame, label, x + round(20 * s), ly, 0.62 * s, MUTED)
        _text(frame, val, x + round(130 * s), ly, 0.62 * s, WHITE)


def _draw_shuttle(frame, row, window, scales, s, x, y, w, h):
    _panel(frame, x, y, w, h)
    _text(frame, "Shuttle Speed", x + round(14 * s), y + round(26 * s), 0.6 * s, MUTED)
    shuttle = row["shuttle"] if row is not None else None
    value = f"{shuttle:.1f} m/s" if shuttle is not None else "--"
    _text_right(frame, value, x + w - round(14 * s), y + round(26 * s), 0.65 * s, WHITE, 2)
    bx, by = x + round(14 * s), y + round(40 * s)
    bw, bh = w - round(28 * s), round(8 * s)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), GRID, -1)
    if shuttle is not None:
        fill = int(bw * min(1.0, shuttle / scales["shuttle"]))
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), SHUTTLE_COLOR, -1)
    _sparkline(frame, [r["shuttle"] if r else None for r in window],
               bx, y + round(58 * s), bw, round(36 * s), SHUTTLE_COLOR, scales["shuttle"], s)


def _sparkline(frame, values, x, y, w, h, color, vmax, s):
    cv2.rectangle(frame, (x, y), (x + w, y + h), GRID, 1)
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        if v is None:
            continue
        px = x + round(i * (w - 1) / max(1, n - 1))
        py = y + h - round(min(1.0, v / vmax) * (h - 1))
        pts.append((px, py))
    if len(pts) >= 2:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color,
                      max(1, round(2 * s)), cv2.LINE_AA)


def _value(row, tid, key):
    if row is None:
        return "--"
    if key == "dist":
        return f"{row[tid]['dist']:.1f} m"
    v = row[tid][key]
    if v is None:
        return "--"
    return f"{v:.1f} m/s" if key == "speed" else f"{v:.1f} m/s2"


def _mmss(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def _panel(frame, x, y, w, h):
    region = frame[max(y, 0):y + h + 1, max(x, 0):x + w + 1]
    cv2.addWeighted(np.full_like(region, PANEL), 0.6, region, 0.4, 0, region)


def _text(frame, text, x, y, size, color=WHITE, thick=1):
    cv2.putText(frame, text, (x, y), FONT, size, color, thick, cv2.LINE_AA)


def _text_center(frame, text, cx, y, size, color=WHITE, thick=1):
    w = cv2.getTextSize(text, FONT, size, thick)[0][0]
    _text(frame, text, cx - w // 2, y, size, color, thick)


def _text_right(frame, text, right_x, y, size, color=WHITE, thick=1):
    w = cv2.getTextSize(text, FONT, size, thick)[0][0]
    _text(frame, text, right_x - w, y, size, color, thick)
