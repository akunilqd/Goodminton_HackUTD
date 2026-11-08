# Badminton Analysis — Modules 1, 2 and 3

Analyze a badminton video with three independent modules and one Python entry point.
Modules 4–6 (hit detection, shot classification and analytics) are outside this implementation.

```text
HackUTD/
├── module1_players/       # YOLO11m → ByteTrack → RTMPose-L, 17 joints
├── module2_shuttle/       # TrackNetV3 → InpaintNet, aligned shuttle positions
├── module3_court/         # 4 court corners → fixed homography → court metres
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
environment. `gpu_half = True` enables FP16 for YOLO player detection, mixed precision
for RTMPose, and FP16 for TrackNet. Set it to `False` for FP32 inference while
keeping CUDA enabled. InpaintNet keeps FP32 coordinates on the GPU. cuDNN kernel
selection accelerates CUDA inference. ByteTrack, video decoding and OpenCV homography
retain their CPU paths.

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
players near the lines.

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
`sys.executable` uses the Python running `main.py`. The adapter supplies the device and
precision from the shared settings. If you use a separate environment for CUDA,
install its CUDA build of PyTorch too.

Download and extract the [official checkpoint archive](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view)
linked in the upstream README. Put these files in `weights/`:

- `TrackNet_best.pt`
- `InpaintNet_best.pt`

The adapter runs TrackNet once and passes the trajectory to InpaintNet in memory.
The models run in a separate process with a configurable timeout, and progress
appears in the terminal. Use `.mp4` input and a new output directory for each run.

### Module 3: court detection and mapping

[CourtKeyNet](https://huggingface.co/Cracked-ANJ/CourtKeyNet) detects four corners
in far-left, far-right, near-right, near-left order. It runs once at
`court_init_frame` and uses the same homography for the whole fixed-camera clip.
If `court_weights` is `None`, the detector uses the local
`weights/courtkeynet_finetuned.safetensors` checkpoint when present or downloads
the pinned weights.

Inspect the corners on a candidate calibration frame:

```powershell
python main.py court-frame data/match.mp4 --frame 0 --output outputs/court-inspection
```

This exports `court.jpg` and `court.json`. Metre coordinates run x across the
6.10 m doubles width and y along the 13.40 m length from the far-left corner.

For a **fixed camera**, use manual calibration without the court ML model:

1. Copy `configs/calibration.example.json` to `configs/my_court.json`.
2. Replace its synthetic pixel coordinates and image dimensions with those from
   your video. Pair at least four non-collinear image points with court metres.
3. Set `court_calibration = Path("configs/my_court.json")` in `config.py`.

The example orders the doubles corners far-left, far-right, near-right, near-left.
Start a new calibration after a camera move, zoom, or cut. Calibration resolution
mismatches and invalid court corners fail with an error.

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

`main.py run` writes one file, `annotated.mp4`, to the selected output directory.
It preserves the source resolution and FPS and contains no audio. When Module 3
is enabled, it draws a top-down mini-court directly into each frame using Fixed H.
Selected modules add player tracks and poses, court keypoints, and a shuttle trail.
The mini-court shows player positions. The shuttle trail appears on the main video.

Tracking and projection results stay in memory. Each run requires a new output
directory. If a run fails, the incomplete video is removed.

## Verification

Previous GPU verification with the legacy 30-point detector used the RTX 3050
6 GB and the existing `.venv-setup` environment
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

FP16 can slightly change numerical predictions; use `gpu_half = False` for FP32.

The core video path streams frames; Module 2 retains coordinate records in memory.
Long videos, occlusions and camera cuts can reduce model/tracker accuracy.
Third-party code and weights retain their own licenses.
