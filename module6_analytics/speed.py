"""Estimate speed at the center of three consecutive video frames."""

import csv
import json
from math import hypot, isfinite
from pathlib import Path

from schemas import Court, FrameResult, Player, Point, Shuttle


def _speed_at_frame(
    previous: FrameResult,
    current: FrameResult,
    following: FrameResult,
    positions: tuple[Point | None, Point | None, Point | None],
) -> float | None:
    if previous.frame_index + 1 != current.frame_index or current.frame_index + 1 != following.frame_index:
        return None

    before, center, after = positions
    if before is None or center is None or after is None:
        return None
    if not all(isfinite(value) for point in (before, center, after) for value in point):
        return None

    dt_before = current.timestamp_s - previous.timestamp_s
    dt_after = following.timestamp_s - current.timestamp_s
    span = dt_before + dt_after
    if dt_before <= 0 or dt_after <= 0 or not isfinite(span):
        return None

    # Weight the two slopes so the derivative is evaluated at the center time,
    # even when the timestamps are unevenly spaced.
    velocity = [
        (dt_after / span) * (p1 - p0) / dt_before
        + (dt_before / span) * (p2 - p1) / dt_after
        for p0, p1, p2 in zip(before, center, after)
    ]
    speed = hypot(*velocity)
    return speed if isfinite(speed) else None


def _frame_from_dict(row: dict) -> FrameResult:
    players = [
        Player(
            track_id=p.get("track_id"),
            bbox_xyxy=tuple(p["bbox_xyxy"]),
            confidence=p.get("confidence", 0.0),
            keypoints=p.get("keypoints", []),
            foot_pixel=tuple(p["foot_pixel"]) if p.get("foot_pixel") else (0.0, 0.0),
            court_xy_m=tuple(p["court_xy_m"]) if p.get("court_xy_m") else None,
        )
        for p in (row.get("players") or [])
    ]
    sh = row.get("shuttle")
    shuttle = None
    if sh:
        shuttle = Shuttle(
            pixel=tuple(sh["pixel"]) if sh.get("pixel") else None,
            raw_pixel=tuple(sh["raw_pixel"]) if sh.get("raw_pixel") else None,
            visible=sh.get("visible", False),
            court_projection_m=tuple(sh["court_projection_m"]) if sh.get("court_projection_m") else None,
        )
    ct = row.get("court")
    court = Court(**ct) if ct else None
    return FrameResult(
        frame_index=row["frame_index"],
        timestamp_s=row["timestamp_s"],
        players=players,
        shuttle=shuttle,
        court=court,
    )


def _shuttle_speed_m_s(
    previous: FrameResult, current: FrameResult, following: FrameResult,
) -> float | None:
    """Estimate shuttlecock speed in m/s using court projection coordinates."""
    def proj(f: FrameResult) -> Point | None:
        return f.shuttle.court_projection_m if f.shuttle else None

    return _speed_at_frame(previous, current, following,
                           (proj(previous), proj(current), proj(following)))


def export_speed_csv(run_dir: Path, output: Path) -> dict:
    """Read frames.jsonl from run_dir and write per-frame speed CSV to output."""
    run_dir = Path(run_dir).resolve()
    output  = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    frames: list[FrameResult] = []
    with (run_dir / "frames.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                frames.append(_frame_from_dict(json.loads(line)))
    frames.sort(key=lambda fr: fr.frame_index)

    track_ids = sorted({
        p.track_id for fr in frames for p in (fr.players or [])
        if p.track_id is not None
    })

    fieldnames = (
        ["frame_index", "timestamp_s", "shuttle_speed_px_s", "shuttle_speed_m_s"]
        + [f"player{tid}_speed_m_s" for tid in track_ids]
    )

    rows_written = 0
    with output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1, len(frames) - 1):
            prev, cur, nxt = frames[i - 1], frames[i], frames[i + 1]
            row: dict = {
                "frame_index": cur.frame_index,
                "timestamp_s": round(cur.timestamp_s, 6),
                "shuttle_speed_px_s": shuttle_speed_px_s(prev, cur, nxt),
                "shuttle_speed_m_s":  _shuttle_speed_m_s(prev, cur, nxt),
            }
            for tid in track_ids:
                row[f"player{tid}_speed_m_s"] = player_speed_m_s(prev, cur, nxt, tid)
            writer.writerow(row)
            rows_written += 1

    return {"output": str(output), "frames": rows_written, "players": track_ids}


def shuttle_speed_px_s(
    previous: FrameResult, current: FrameResult, following: FrameResult,
) -> float | None:
    """Estimate image speed in px/s from observed TrackNet points."""
    return _speed_at_frame(
        previous,
        current,
        following,
        (
            previous.shuttle.raw_pixel if previous.shuttle is not None else None,
            current.shuttle.raw_pixel if current.shuttle is not None else None,
            following.shuttle.raw_pixel if following.shuttle is not None else None,
        ),
    )


def player_speed_m_s(
    previous: FrameResult, current: FrameResult, following: FrameResult, track_id: int,
) -> float | None:
    """Estimate ground speed in m/s for one player from projected foot positions."""
    def court_position(frame: FrameResult) -> Point | None:
        return next(
            (player.court_xy_m for player in frame.players or []
             if player.track_id is not None and player.track_id == track_id),
            None,
        )

    return _speed_at_frame(
        previous,
        current,
        following,
        (court_position(previous), court_position(current), court_position(following)),
    )
