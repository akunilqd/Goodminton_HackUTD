"""Stable, JSON-compatible contracts shared by the independent modules."""

from dataclasses import dataclass, field

Point = tuple[float, float]


@dataclass
class Player:
    track_id: int | None
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: list[tuple[float, float, float]]
    foot_pixel: Point
    court_xy_m: Point | None = None
    on_court: bool | None = None


@dataclass
class Shuttle:
    pixel: Point | None
    raw_pixel: Point | None
    visible: bool
    # Upstream exports visibility, not a probability. Do not fabricate confidence.
    confidence: float | None = None
    rectified: bool = False
    source: str = "missing"
    court_projection_m: Point | None = None


@dataclass
class Court:
    keypoints: list[tuple[float, float, float]] = field(default_factory=list)
    homography: list[list[float]] | None = None
    inlier_count: int = 0
    reprojection_error_m: float | None = None
    status: str = "not_detected"


@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float
    players: list[Player] | None = None
    shuttle: Shuttle | None = None
    court: Court | None = None


@dataclass
class HitEvent:
    event_id: int
    rally_id: str | int | None
    frame_index: int
    timestamp_s: float
    hitter_id: int | None
    hit_score: float  # Heuristic evidence score, NOT a calibrated probability.
    hitter_score: float
    timing_uncertainty_frames: int
    quality_flags: list[str] = field(default_factory=list)
    source: str = "heuristic_v1"


@dataclass
class ShotEvent:
    shot_id: int
    start_frame: int          # Label display begins at this hit.
    end_frame: int            # Inclusive label display end, separate from model context.
    hitter_id: int | None
    player_side: str | None   # "Top" or "Bottom" based on court position
    class_id: int             # 0..5: Smash, Clear, Drop, Drive, Net, Serve
    shot_type_raw: str        # Native 18-way argmax, e.g. "Bottom-Net-Kill".
    shot_type_en: str         # Winning group; may differ from the native argmax group.
    confidence: float         # Sum of native softmax probabilities in the winning group.
    source: str = "bst_badmintondb_18_to_6_v1"
    raw_class_id: int | None = None
    raw_confidence: float | None = None
    class_probabilities: dict[str, float] = field(default_factory=dict)
    low_confidence: bool = False
    quality_flags: list[str] = field(default_factory=list)
    input_start_frame: int | None = None  # Inclusive BST context bounds.
    input_end_frame: int | None = None
    input_sample_count: int | None = None  # After temporal sampling, before zero-padding.
