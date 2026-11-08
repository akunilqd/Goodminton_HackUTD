"""Estimate speed at the center of three consecutive video frames."""

from math import hypot, isfinite

from schemas import FrameResult, Point


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
