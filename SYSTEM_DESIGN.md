# Badminton Analysis System — System Design

## Pipeline Overview

```
Match Video
│
├── [M1] Player Detection & Tracking
│         YOLO11 → ByteTrack → RTMPose-L
│
├── [M2] Shuttle Tracking
│         TrackNetV3 → InpaintNet
│
├── [M3] Court Detection & 2D Mapping
│         Court Keypoint Model → OpenCV Homography
│
└─────────────────────┬─────────────────────
                      ↓
             [M4] Hit Detection
                      ↓
             [M5] Shot Classification
             Smash / Clear / Drop / Drive / Lift / Serve
                      ↓
             [M6] Analytics
             ├── 2D court map
             ├── Player heatmap
             ├── Shuttle trajectory
             ├── Estimated speed
             ├── Shot statistics
             └── Rally timeline
```

---

## Module Table

| # | Module | Task | Model / Tool | Usage | Pretrained Weights / Link |
|---|--------|------|--------------|-------|--------------------------|
| 1 | Player Detection & Tracking | Object detection | YOLO11m | Pretrained | Auto-downloaded via `ultralytics` pip; `yolo11m.pt` |
| 1 | Player Detection & Tracking | Multi-object tracking | ByteTrack | Pretrained | Bundled in Ultralytics — `tracker=bytetrack.yaml` |
| 1 | Player Pose Estimation | 2D pose keypoints | RTMPose-L | Pretrained | See Module 1 weights below |
| 2 | Shuttle Tracking | Ball trajectory detection | TrackNetV3 | Pretrained (fine-tune if needed) | See Module 2 weights below |
| 2 | Shuttle Tracking | Trajectory gap filling | InpaintNet | Pretrained (fine-tune if needed) | Bundled with TrackNetV3 release |
| 3 | Court Detection | Keypoint detection on court | CourtKeyNet-Finetuned (once per video) | Pretrained | https://huggingface.co/Cracked-ANJ/CourtKeyNet |
| 3 | Court 2D Mapping | Perspective transform | OpenCV Homography | No weights (geometry) | `cv2.findHomography` + `cv2.perspectiveTransform` |
| 4 | Hit Detection | Detect shuttle–racket contact frames | Heuristic / 1D CNN | Train from scratch | ShuttleSet annotations |
| 5 | Shot Classification | Classify shot type | BST (Badminton Shot Transformer) | Pretrained | See Module 5 weights below |
| 6 | Analytics | Speed, heatmap, stats, timeline | Geometry + NumPy | No weights | Pure computation |

---

## Pretrained Weights & Repositories

### Module 1 — YOLO11 + ByteTrack + RTMPose-L

**YOLO11 (Ultralytics)**
- Repo: https://github.com/ultralytics/ultralytics
- Install: `pip install ultralytics`
- Weights auto-download on first run, or manually:
  ```
  yolo11n.pt  →  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
  yolo11m.pt  →  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
  yolo11l.pt  →  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt
  ```
- Recommended for this project: `yolo11m.pt` (balance of speed and accuracy)

**ByteTrack**
- Integrated into Ultralytics — no separate install required
- Config: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml
- Usage: `model.track(source, tracker="bytetrack.yaml")`

**RTMPose-L**
- Repo: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- Install: `pip install mmpose mmcv mmdet`
- Weights:
  ```
  rtmpose-l (body, 256×192):
  https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth

  rtmpose-l (body, 384×288, higher res):
  https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth
  ```
- Config files: https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose

---

### Module 2 — TrackNetV3 + InpaintNet

**TrackNetV3**
- Repo: https://github.com/qaz812345/TrackNetV3
- Weights (`TrackNet_best.pt` + `InpaintNet_best.pt`):
  - Download through the checkpoint link in the inference instructions: https://github.com/qaz812345/TrackNetV3#inference
  - Or from the repo's Google Drive link in the README
- Alternative / original TrackNet repos:
  - TrackNetV2: https://github.com/ncts-badminton/TrackNetV2
  - MonoTrack (badminton): https://github.com/chang12310/Badminton-2023

**InpaintNet**
- Bundled inside the TrackNetV3 repository (same weights release)
- Fills in occluded/missing shuttle positions in the trajectory

---

### Module 3 - CourtKeyNet (fixed camera)

- Model/weights: https://huggingface.co/Cracked-ANJ/CourtKeyNet
- Source: https://github.com/adithyanraj03/CourtKeyNet (MIT).
- Pinned fine-tuned safetensors checkpoint; architecture embedded in detector.py.
- Four outer corners TL, TR, BR, BL map to the full 6.10 x 13.40 m court.
- Infer once at `court_init_frame` in FP32, validate geometry, then cache corners/H.
- M1 uses the same fixed hull; M5 can project ankles using H.
- Invalid calibration fails; camera movement/cuts require recalibration.
- Validate with internal court lines, not just the four-point fit residual.
- Manual calibration remains available. No 30-point template is needed.

---

### Module 4 — Hit Detection

- **Approach:** Heuristic (shuttle velocity spike near player wrist keypoint) or 1D CNN over shuttle trajectory + pose time series
- **No pretrained weights** — trained from scratch on ShuttleSet rally annotations
- Training dataset: ShuttleSet — https://github.com/wywywywy/ShuttleSet

---

### Module 5 — Shot Classification (BST)

**BST — Badminton Shot Transformer**
- Repo: https://github.com/wywywywy/bst-badminton
- Weights: available in repo releases or linked in README
- Input: sequence of shuttle positions + player pose keypoints around a hit event
- Output: shot type — `Smash | Clear | Drop | Drive | Lift | Serve`
- Training / evaluation dataset:
  - ShuttleSet: https://github.com/wywywywy/ShuttleSet
  - ShuttleSet22: https://github.com/wywywywy/ShuttleSet22

---

### Module 6 — Analytics (no ML weights)

All analytics are computed from outputs of Modules 1–5 using standard Python libraries:

| Output | Method |
|--------|--------|
| 2D court map | Project player + shuttle 2D coords onto court template image |
| Player heatmap | `seaborn.kdeplot` or `matplotlib` over 2D court positions |
| Shuttle trajectory | Draw polyline of (x, y) positions with speed color-coding |
| Estimated speed | `speed = pixel_displacement / frame_Δt × real_world_scale` |
| Shot statistics | Count per shot type per player, win/loss rate per shot |
| Rally timeline | Horizontal bar chart: rally segments with shot labels |

---

## Data Flow

```
Video frames
    │
    ├─[M1]─► Player bounding boxes + track IDs (per frame)
    │         └─[M1]─► Pose keypoints (17 joints per player per frame)
    │
    ├─[M2]─► Shuttle (x, y) per frame + confidence
    │         └─[M2]─► Inpainted trajectory (gap-filled)
    │
    ├─[M3]─► Court corners (4 points, cached)
    │         └─[M3]─► Homography matrix H (image → 2D court)
    │
    └─────────────────────────────────────────────────────────────────
                                  │
    [M4] Hit Detection ◄──────────┤ shuttle trajectory + pose wrist
          │ hit_frame, hitter_id  │
    [M5] Shot Classification ◄────┘ pose sequence around hit + shuttle
          │ shot_type label
    [M6] Analytics ◄──────────────── all above merged
          │
    ► Heatmap PNG / Rally Timeline PNG / Stats JSON / Annotated video
```

---

## Environment & Dependencies

```bash
# Core
Python >= 3.10
torch >= 2.0
torchvision

# YOLO11 + ByteTrack
pip install ultralytics

# RTMPose (MMPose stack)
pip install openmim
mim install mmengine mmcv mmdet mmpose

# Computer vision / analytics
pip install opencv-python numpy matplotlib seaborn pandas

# TrackNetV3 (clone separately)
git clone https://github.com/qaz812345/TrackNetV3
```

---

## Dataset Sources

| Purpose | Dataset | Link |
|---------|---------|------|
| Player detection fine-tune | Roboflow badminton player dataset | https://universe.roboflow.com (search "badminton player") |
| Shuttle tracking | TrackNetV3 dataset | https://github.com/qaz812345/TrackNetV3 |
| Court keypoints | Bundled with court model | See Module 3 |
| Shot classification + hit detection | ShuttleSet | https://github.com/wywywywy/ShuttleSet |
| Shot classification (extended) | ShuttleSet22 | https://github.com/wywywywy/ShuttleSet22 |

---

## Quick-Start Verification Checklist

- [ ] Run YOLO11 on a 30s clip — confirm player bounding boxes appear
- [ ] Run ByteTrack — confirm consistent player IDs across frames
- [ ] Run RTMPose — confirm 17 keypoints visible on each player
- [ ] Run TrackNetV3 — confirm shuttle (x,y) CSV output aligns with timestamps
- [ ] Run InpaintNet — confirm trajectory gaps are filled
- [ ] Run court detector — confirm 4 corners and projected internal lines align
- [ ] Verify homography by projecting corners to 2D top-down template
- [ ] Run hit detector — confirm hit events flagged near wrist + shuttle proximity
- [ ] Run BST — confirm shot labels from {Smash, Clear, Drop, Drive, Lift, Serve}
- [ ] Generate analytics: heatmap PNG, rally timeline PNG, stats JSON


## Modules 1-3 Implementation

- `module1_players/`: YOLO11m person detection, persistent ByteTrack IDs, and
  top-down RTMPose-L with 17 COCO joints per box.
- `module2_shuttle/`: separate-process integration with qaz812345/TrackNetV3,
  including InpaintNet, raw/repaired CSV alignment, and explicit missing points.
  The similarly named alenzenx project does not provide this InpaintNet interface.
- `module3_court/`: one-shot 4-corner CourtKeyNet and homography with validated
  image-to-court correspondences; manual fixed-camera calibration is also supported.
- `main.py` selects modules; `pipeline.py` joins outputs by zero-based frame index.
  See `README.md` and `config.py` for setup, settings and commands.

The supplied RTMPose default uses the official COCO RTMPose-L config/checkpoint
pair. The Body7 checkpoint above requires its matching config if selected.
CourtKeyNet maps four outer corners to known court metres; no template is needed.
Shuttle confidence is null because the upstream CSV exports visibility only.
A shuttle homography projection is a floor-plane approximation, not a 3D position.
The implementation currently stops at Module 3.
