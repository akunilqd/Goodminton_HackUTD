"""Edit the Settings defaults below to configure the analysis pipeline."""

import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_COURT_WEIGHTS = PROJECT_DIR / "weights/courtkeynet_finetuned.safetensors"


@dataclass(frozen=True)
class Settings:
    # Input / Output
    video: Path = Path("data/match178_000.mp4")
    output: Path = Path("outputs/run3")
    modules: tuple = ("1", "2", "3", "4", "5")

    # Module 1: YOLO11 + ByteTrack + RTMPose-L.
    device: str = "cuda:0"  # Use "cpu" on a machine without CUDA-enabled PyTorch.
    gpu_half: bool = True  # FP16 neural inference on CUDA; geometry stays FP32/FP64.
    cpu_workers: int = 8  # Threads preparing shuttle frames; changes speed, not inputs.
    player_batch_size: int = 8  # Frames per YOLO/RTMPose forward pass.
    player_weights: str = "weights/yolo11m.pt"
    player_confidence: float = 0.25
    pose_confidence: float = 0.3
    # Keep the full MMPose checkout: this config imports relative base files.
    rtmpose_config: Path | None = Path(
        "external/mmpose/configs/body_2d_keypoint/rtmpose/coco/"
        "rtmpose-l_8xb256-420e_coco-256x192.py"
    )
    rtmpose_weights: Path | None = Path("weights/rtmpose-l.pth")

    # Module 2: use another environment's Python executable if needed.
    tracknet_repo: Path = Path("external/TrackNetV3")
    tracknet_python: str = sys.executable
    tracknet_weights: Path = Path("weights/TrackNet_best.pt")
    inpaintnet_weights: Path = Path("weights/InpaintNet_best.pt")
    tracknet_batch_size: int = 8
    tracknet_timeout_seconds: int = 3600

    # Module 3: None uses the local CourtKeyNet checkpoint or downloads it.
    court_weights: str | None = None
    court_init_frame: int = 0
    court_keypoint_confidence: float = 0.5
    court_ransac_threshold_m: float = 0.15
    court_max_error_m: float = 0.2
    court_calibration: Path | None = None  # JSON: manual fixed-camera calibration.

    # Module 5: BST shot classification.
    bst_repo: Path = Path("external/BST-Badminton-Stroke-type-Transformer")
    bst_weights: Path = Path("weights/bst_CG_AP_JnB_bone.pt")
    bst_seq_len: int = 72
    bst_batch_size: int = 64
    bst_model: str = "BST_CG_AP"
    bst_min_seg_frames: int = 15      # Flag shorter input windows; still infer with padding.
    bst_conf_threshold: float = 0.2  # Flag low confidence; always retain the winning class.
    # Experimental context window for the current BadmintonDB checkpoint.
    bst_context_limit_s: float = 1.5
    bst_context_fallback_s: float = 0.5  # No neighboring hit in the same rally.
    bst_context_after_next_s: float = 0.25

    # Module 4: offline CPU heuristics; velocities are image diagonals / second.
    hit_max_gap_s: float = 0.08
    hit_velocity_window_s: float = 0.16
    hit_wrist_window_s: float = 0.08
    hit_merge_window_s: float = 0.12
    hit_min_speed: float = 0.04
    hit_min_velocity_change: float = 0.12
    hit_turn_degrees: float = 40.0
    hit_wrist_radius: float = 0.65  # Distance / player bbox diagonal (racket reach).
    hit_hitter_margin: float = 0.15
    hit_min_score: float = 0.60

    def __post_init__(self):
        if isinstance(self.court_init_frame, bool) or not isinstance(self.court_init_frame, int) or self.court_init_frame < 0:
            raise ValueError("court_init_frame must be a non-negative integer")
        # Resolve paths from this file, even when launched from another directory.
        for name in (
            "video",
            "output",
            "rtmpose_config",
            "rtmpose_weights",
            "tracknet_repo",
            "tracknet_weights",
            "inpaintnet_weights",
            "court_calibration",
            "bst_repo",
            "bst_weights",
        ):
            value = getattr(self, name)
            if value is not None:
                path = Path(value).expanduser()
                object.__setattr__(self, name, path if path.is_absolute() else PROJECT_DIR / path)
        for name in ("player_weights", "court_weights", "tracknet_python"):
            value = getattr(self, name)
            if value and (name == "court_weights" or "/" in value or "\\" in value):
                path = Path(value).expanduser()
                object.__setattr__(
                    self, name, str(path if path.is_absolute() else PROJECT_DIR / path)
                )
        for name in (
            "player_confidence",
            "pose_confidence",
            "court_keypoint_confidence",
            "hit_min_score",
            "hit_hitter_margin",
        ):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in (
            "cpu_workers",
            "player_batch_size",
            "bst_batch_size",
            "tracknet_batch_size",
            "tracknet_timeout_seconds",
            "court_ransac_threshold_m",
            "court_max_error_m",
            "hit_max_gap_s",
            "hit_velocity_window_s",
            "hit_wrist_window_s",
            "hit_merge_window_s",
            "hit_min_speed",
            "hit_min_velocity_change",
            "hit_wrist_radius",
            "bst_context_limit_s",
            "bst_context_fallback_s",
            "bst_context_after_next_s",
        ):
            value = getattr(self, name)
            if not 0 < value < float("inf"):
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.hit_turn_degrees < 180:
            raise ValueError("hit_turn_degrees must be in (0, 180)")
        if not isinstance(self.bst_seq_len, int) or self.bst_seq_len < 1:
            raise ValueError("bst_seq_len must be a positive integer")
        if not isinstance(self.bst_min_seg_frames, int) or self.bst_min_seg_frames < 1:
            raise ValueError("bst_min_seg_frames must be a positive integer")
        if self.bst_model not in {"BST", "BST_CG", "BST_AP", "BST_CG_AP"}:
            raise ValueError("bst_model must be one of BST, BST_CG, BST_AP, BST_CG_AP")
        if not 0 < self.bst_conf_threshold <= 1:
            raise ValueError("bst_conf_threshold must be in (0, 1]")


def require_file(path: Path | None, setting: str) -> Path:
    if path is None or not path.is_file():
        raise ValueError(f"{setting} must point to an existing file; got {path!s}. See README.md.")
    return path.resolve()
