"""Edit the Settings defaults below to configure all three modules."""

import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_COURT_WEIGHTS = PROJECT_DIR / "weights/courtkeynet_finetuned.safetensors"


@dataclass(frozen=True)
class Settings:
    # Input / Output
    video: Path = Path("data/match178_000.mp4")
    output: Path = Path("outputs/run2")
    modules: tuple = ("1", "2", "3")

    # Module 1: YOLO11 + ByteTrack + RTMPose-L.
    device: str = "cuda:0"  # Use "cpu" on a machine without CUDA-enabled PyTorch.
    gpu_half: bool = True  # FP16 neural inference on CUDA; geometry stays FP32/FP64.
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

    def __post_init__(self):
        if type(self.court_init_frame) is not int or self.court_init_frame < 0:
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
        ):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in (
            "tracknet_batch_size",
            "tracknet_timeout_seconds",
            "court_ransac_threshold_m",
            "court_max_error_m",
        ):
            value = getattr(self, name)
            if not 0 < value < float("inf"):
                raise ValueError(f"{name} must be positive and finite")


def require_file(path: Path | None, setting: str) -> Path:
    if path is None or not path.is_file():
        raise ValueError(f"{setting} must point to an existing file; got {path!s}. See README.md.")
    return path.resolve()
