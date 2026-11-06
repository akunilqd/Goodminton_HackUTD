# Badminton Analysis — Modules 1, 2 and 3

Analyze a badminton video with three independent modules and one Python entry point.
Modules 4–6 (hit detection, shot classification and analytics) are outside this implementation.

```text
HackUTD/
├── module1_players/       # YOLO11m → ByteTrack → RTMPose-L, 17 joints
├── module2_shuttle/       # TrackNetV3 → InpaintNet, aligned shuttle positions
├── module3_court/         # 30 court keypoints → homography → court metres
├── configs/              # Example manual court calibration
├── main.py               # Run, check setup, inspect court keypoints
├── pipeline.py           # Join module outputs by frame and write results
├── schemas.py            # Shared player, shuttle, court and frame records
├── video.py              # Streaming video reader
├── config.py             # All settings, grouped by module
├── requirements.txt
├── .gitignore
└── SYSTEM_DESIGN.md
```

## Setup

Use Python 3.10 for the full MMPose stack. Python 3.10–3.11 can run the pinned core pipeline;
Python 3.14 is not supported by this project's ML dependency setup.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`. Edit the `Settings`
defaults in `config.py` to configure the project. No API keys are required.
Relative configuration paths resolve from the project directory. Set
`device = "cuda:0"` (the default) for all three modules. CUDA must be available in
both the main Python and `tracknet_python`; unavailable CUDA fails with an error
instead of silently running on CPU. Set `device = "cpu"` explicitly for CPU runs.
Optional paths use
`None`; set a path with `Path("weights/model.pth")`. Environment variables do not
override these settings.

### GPU environment (Windows, Python 3.10)

The existing workspace environment is `.venv-setup`. Install the CUDA builds before
the other requirements, keeping PyTorch and MMCV compatible:

```powershell
.\.venv-setup\Scripts\python.exe -m pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
.\.venv-setup\Scripts\python.exe -m pip install --force-reinstall --no-deps https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv-2.1.0-cp310-cp310-win_amd64.whl
.\.venv-setup\Scripts\python.exe -B main.py check
.\.venv-setup\Scripts\python.exe -B main.py run
```

The CUDA wheel pairing follows the [PyTorch version table](https://docs.pytorch.org/get-started/previous-versions/).
`check` reports each selected module's device and checks Module 2 in its own Python
environment. `gpu_half = True` enables FP16 for both YOLO models, mixed precision
for RTMPose, and FP16 for TrackNet. Set it to `False` for FP32 inference while
keeping CUDA enabled. InpaintNet keeps FP32 coordinates on the GPU. cuDNN kernel
selection and pinned TrackNet batches accelerate CUDA inference. ByteTrack,
video decoding, CSV/JSON output and OpenCV homography retain their CPU paths.

### Module 1: player detection, tracking and pose

YOLO11m weights download automatically on first use. ByteTrack is bundled with
Ultralytics. RTMPose requires a compatible MMPose/MMCV/PyTorch installation:

```powershell
python -m pip install openmim
mim install "mmengine>=0.9,<1"
mim install "mmcv>=2.0.1,<2.2"
mim install "mmdet>=3.1,<3.4"
mim install "mmpose>=1.3,<1.4"
git clone --depth 1 https://github.com/open-mmlab/mmpose.git external/mmpose
New-Item -ItemType Directory -Force weights
Invoke-WebRequest -Uri "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth" -OutFile weights/rtmpose-l.pth
```

MMCV includes compiled extensions. Choose a PyTorch/CUDA combination supported by
its wheels before installing it; a source build may otherwise be required. Follow
the [MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html)
for your platform. The GPU commands above match this workspace's Python 3.10 stack.

The supplied `config.py` pairs the official RTMPose-L COCO config with its matching
checkpoint from the [model table](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose_coco.md).
It produces 17 COCO joints. To use the design's Body7 checkpoint instead, change
both `rtmpose_config` and `rtmpose_weights` to a matching Body7 pair. Keep the complete
MMPose checkout because configs import relative base files.

M1 uses M3 court keypoints to keep only the two singles players. Run M1 together
with M3 (`--modules 1 3`, or all three modules). People outside the court are
excluded before pose inference; a small border tolerance retains established
players near the lines. This filter works without a metre-coordinate template.

The first pair receives ID 1 (initially far) and ID 2 (initially near). These IDs
remain reserved until the video is reset. Matching uses ByteTrack IDs and the
last foot positions only; it never renumbers players by their current side.
Missing players are omitted until detected again, rather than drawing stale
boxes. Missing court detections do not fall back to tracking spectators.
Long occlusions, camera cuts or ambiguous crossings can still cause incorrect
identity matching; this is a simple singles tracker, not a person ReID model.

### Module 2: shuttle tracking and gap filling

The original design linked a different project called TrackNetV3. This adapter uses
[qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3), which supplies the
requested TrackNet and InpaintNet networks.

```powershell
git clone https://github.com/qaz812345/TrackNetV3.git external/TrackNetV3
```

Install its dependencies following that repository's instructions, preferably in a
separate environment because its published dependency pins are older. Set
`tracknet_python` to that environment's Python executable. The default
`sys.executable` uses the Python running `main.py`. The local inference script accepts
`--device` and `--half`, which the adapter supplies from the shared settings.
If you use another Python environment, install its CUDA build of PyTorch too.
Keep these local changes to `external/TrackNetV3/predict.py` when updating upstream.

Download and extract the [official checkpoint archive](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view)
linked in the upstream README. Put these files in `weights/`:

- `TrackNet_best.pt`
- `InpaintNet_best.pt`

The adapter runs the upstream CLI twice: once for raw positions and once with
InpaintNet. This preserves both trajectories and makes corrections inspectable,
at the cost of running TrackNet twice. It uses non-overlapping temporal windows
and upstream large-video mode. Each subprocess has a configurable timeout and
its own log. Use `.mp4` input and a new output directory for each run.

### Module 3: court detection and mapping

The [court model](https://huggingface.co/Huydinh1205/court_yolo11n_pose) downloads
automatically when `court_weights` is `None`. It produces **30 keypoints**, correcting
the design's older references to 14.

For automatic metre mapping, set `court_template` to a JSON file containing
`court_points_m`: 30 distinct `[x, y]` pairs in the model's exact keypoint order.
The model card describes a snake layout but does not publish the full coordinate
table. No guessed table is enabled. With no template, detection still works and
returns `status="template_required"`, with mapping fields set to `null`.

Inspect numbered detections to establish the correspondence using the training
annotations:

```powershell
python main.py court-frame data/match.mp4 --frame 0 --output outputs/court-inspection
```

This exports `court.jpg` and `court.json`. Use court-floor points only. Metre
coordinates run x across the 6.10 m doubles width and y along the 13.40 m length;
use a consistent origin and orientation.

For a **fixed camera**, use manual calibration without the court ML model:

1. Copy `configs/calibration.example.json` to `configs/my_court.json`.
2. Replace its synthetic pixel coordinates and image dimensions with those from
   your video. Pair at least four non-collinear image points with court metres.
3. Set `court_calibration = Path("configs/my_court.json")` in `config.py`.

The example orders the doubles corners far-left, far-right, near-right, near-left.
Manual calibration must be redone after a camera move, zoom, or cut. The automatic
detector recalculates geometry each frame and does not reuse a stale fit when
detection fails. RANSAC rejects outliers; insufficient or degenerate geometry
produces a status and no homography. Calibration resolution mismatches fail.

## Run

Check local assets and package availability first (this does not run inference):

```powershell
python main.py check
python main.py run data/match.mp4 --output outputs/match-01
```

Run just the modules you need:

```powershell
python main.py check --modules 1 3
python main.py run data/match.mp4 --modules 1 3 --output outputs/players
python main.py run data/match.mp4 --modules 2 --output outputs/shuttle
python main.py run data/match.mp4 --modules 3 --output outputs/court
python main.py run data/match.mp4 --modules 1 3 --output outputs/players-and-court
```

Change settings directly in `config.py`, then rerun the command.

## Output

| File | Contents |
| --- | --- |
| `frames.jsonl` | One JSON record per decoded frame: index, timestamp, players, shuttle, court |
| `metadata.json` | Video dimensions/FPS, enabled modules, processed frame count and run status |
| `shuttle.mp4` / `annotated.mp4` | Annotated video: player boxes + track IDs, RTMPose-L joints/skeleton, numbered court keypoints, and shuttle trail when M2 is enabled |
| `shuttle.csv` | Raw/repaired pixel positions, timestamps, visibility, provenance and court projection |
| `shuttle/raw/` | Upstream TrackNet CSV and inference log |
| `shuttle/inpainted/` | Upstream TrackNet + InpaintNet CSV and inference log |

The annotated video is `shuttle.mp4` when M2 is enabled (preserving the existing
video filename), or `annotated.mp4` when running M1/M3 without M2. It preserves
source resolution and FPS and contains no audio. Overlays use the existing
`pose_confidence` and `court_keypoint_confidence` thresholds. Court points are
labelled C0?C29; no guessed court-line connections are drawn. Inference always
receives the original frame before any drawing.

Module 2 files are only written when selected. Disabled module fields are `null`;
an enabled player module with no detections returns `[]`. Keypoints are
`[x_pixels, y_pixels, confidence]`. Player keypoints use COCO order: nose, eyes,
ears, shoulders, elbows, wrists, hips, knees, ankles, with left before right.
Boxes use `[x1, y1, x2, y2]`. Frames are zero-based; timestamps are `frame_index/fps`.
Use constant-frame-rate footage for accurate timing.

Missing shuttle coordinates remain `null`. Upstream exports binary visibility,
not a probability, so shuttle `confidence` is `null` (blank in CSV). `rectified`
means the final exported coordinate differs from the raw one; it is not the
upstream network's internal repair mask. `source` is `tracknet`, `inpaintnet` or
`missing` based on that comparison.

Player court positions approximate ground contact. An airborne shuttle's
`court_projection_m` is a projection onto the floor plane, **not its physical
3D location or a reliable speed measurement**.

Runs never overwrite an existing output directory. Failed runs retain
`frames.partial.jsonl` when available and mark metadata as failed; a complete
`frames.jsonl` is published only after frame alignment succeeds.

## Verification

GPU verification on the RTX 3050 6 GB used the existing `.venv-setup` environment
(PyTorch 2.1.0+cu118, torchvision 0.16.0+cu118, MMCV 2.1.0 built for CUDA 11.8):

- `main.py check` reports `cuda:0` for all three modules; `pip check` passes.
- Before court filtering, M1/M3 inference passed on a real match frame: 8 people, 17 joints per person,
  and 30 court keypoints. The actual inference models were verified on CUDA;
  both YOLO predictors used FP16. Explicit CPU inference also passed.
- Before court filtering, after two warm-up iterations, five repeated runs of that frame averaged
  75.64 ms for M1+M3 on GPU versus 1008.51 ms on CPU. This is a small inference
  comparison, not an end-to-end video benchmark or an accuracy evaluation.
- Both M2 passes completed the existing 250-frame clip, with CUDA TrackNet and
  InpaintNet, matching frame indices, and about 1499 MiB peak allocated VRAM.
- Missing CUDA, an invalid GPU index, CPU selection, and subprocess device/precision
  propagation were checked. Verification scripts and prediction exports stayed
  in memory; no test files or result artifacts were added.

The simple two-player tracker was also checked on all 250 clip frames: two players
were returned on 247 frames, one on 3 frames, with no far/near identity swaps in
this clip. Reordered detections, changed ByteTrack IDs, short loss/recovery,
missing court detections and reset were checked in memory without creating files.

A verified court template or manual calibration is still required for homography.
FP16 can slightly change numerical predictions; use `gpu_half = False` for FP32.

The core video path streams frames; Module 2 retains coordinate records in memory.
Long videos, occlusions and camera cuts can reduce model/tracker accuracy.
Third-party code and weights retain their own licenses; the court model and
Ultralytics use AGPL-3.0 terms.
