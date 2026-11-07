# Badminton Analysis — Modules 1, 2 and 3

Analyze a badminton video with three independent modules and one Python entry point.
Modules 4–6 (hit detection, shot classification and analytics) are outside this implementation.

```text
HackUTD/
├── module1_players/       # YOLO11m → ByteTrack → RTMPose-L, 17 joints
├── module2_shuttle/       # TrackNetV3 → InpaintNet, aligned shuttle positions
├── module3_court/         # 4 CourtKeyNet corners → homography → court metres
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
environment. `gpu_half = True` enables FP16 for the M1 YOLO model, mixed precision
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

### Module 3: CourtKeyNet fixed-camera calibration

M3 uses [CourtKeyNet-Finetuned](https://huggingface.co/Cracked-ANJ/CourtKeyNet).
The architecture is embedded in `module3_court/detector.py` with its MIT notice;
no extra source files/checkout are needed. Safetensors loads the pinned weights
(revision `8254e285ed0ed496a96fa4632111c80ffb645998`) into a system temporary cache.
Set `court_weights` to a local `.safetensors` path for persistent storage elsewhere.

The pipeline infers once at `court_init_frame` (default 0) in FP32, validates the
four corners, then caches them and H for the full video. Corners are TL, TR, BR, BL,
mapped to `(0,0), (6.10,0), (6.10,13.40), (0,13.40)` metres. No 30-point template
is needed. M1 still uses the fixed corner hull to filter players. `gpu_half` does
not change M3's FP32 calibration. Resolution changes fail; camera moves, zooms
or cuts require recalibration and are not automatically detected.

The repeated score in each corner is a shared court quality heuristic using the
upstream GUI weighting: 0.4 peak + 0.4 valid geometry + 0.2 inverse entropy.
It is not a calibrated per-corner probability. Invalid geometry or low quality
fails explicitly; select another frame or set `court_calibration` to verified
manual correspondences. Four-point fit residual is NOT independent accuracy:
inspect projected internal court lines or measure held-out line intersections.

```powershell
python -B main.py check --modules 3
python -B main.py court-frame data/match178_000.mp4 --frame 0 --output outputs/court-inspection-new
python -B main.py run data/match178_000.mp4 --modules 3 --court-init-frame 0 --output outputs/court-only-new
python -B main.py run data/match178_000.mp4 --modules 1 2 3 --court-init-frame 0 --output outputs/full-new
```

These existing commands create normal artifacts when invoked. `court-frame` and
the pipeline use the same corner/court-line renderer. Run metadata records the
calibration frame and number of model inferences.

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
labelled C0-C3; standard court-floor lines are projected through the homography. Inference always
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

Current CourtKeyNet verification (2026-11-07): strict checkpoint loading and CUDA
FP32 inference passed on `match178_000.mp4`. The 250-frame clip was decoded and
processed with one model inference and a cached four-corner homography. The actual
`main.run`/pipeline path was also exercised with output sinks mocked in memory,
including 250 rendered frames, to avoid creating extra files. Geometry and visual
alignment are not ground-truth accuracy measurements; the preview shows some
boundary offset and no independently annotated accuracy score is available.


The older YOLO court benchmarks below are historical, not current M3 measurements.

GPU verification on the RTX 3050 6 GB used the existing `.venv-setup` environment
(PyTorch 2.1.0+cu118, torchvision 0.16.0+cu118, MMCV 2.1.0 built for CUDA 11.8):

- `main.py check` reports `cuda:0` for all three modules; `pip check` passes.
- Before court filtering, M1/M3 inference passed on a real match frame: 8 people, 17 joints per person,
  and 4 CourtKeyNet corners. The actual inference models were verified on CUDA;
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

CourtKeyNet supplies four-corner homography without a separate court template.
FP16 can slightly change numerical predictions; use `gpu_half = False` for FP32.

The core video path streams frames; Module 2 retains coordinate records in memory.
Long videos, occlusions and camera cuts can reduce model/tracker accuracy.
Third-party code and weights retain their own licenses. CourtKeyNet uses MIT;
Ultralytics in M1 uses AGPL-3.0 terms.
