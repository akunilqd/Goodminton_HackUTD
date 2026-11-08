# Badminton Analysis — Modules 1–5

Analyze a badminton video with five modules and one Python entry point.
M5 uses BST BadmintonDB and exposes six basic shot classes.

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

```powershelll
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

### Module 3: CourtKeyNet fixed-camera calibration

[CourtKeyNet](https://huggingface.co/Cracked-ANJ/CourtKeyNet) detects four corners
in far-left, far-right, near-right, near-left order. It runs once at
`court_init_frame` and uses the same homography for the whole fixed-camera clip.
If `court_weights` is `None`, the detector uses the local
`weights/courtkeynet_finetuned.safetensors` checkpoint when present or downloads
the pinned weights.

Inspect the corners on a candidate calibration frame:

```powershell
python -B main.py check --modules 3
python -B main.py court-frame data/match178_000.mp4 --frame 0 --output outputs/court-inspection-new
python -B main.py run data/match178_000.mp4 --modules 3 --court-init-frame 0 --output outputs/court-only-new
python -B main.py run data/match178_000.mp4 --modules 1 2 3 --court-init-frame 0 --output outputs/full-new
```

This exports `court.jpg` and `court.json`. Metre coordinates run x across the
6.10 m doubles width and y along the 13.40 m length from the far-left corner.

### Module 4: offline hit detection (CPU)

`module4/` consumes aligned M1 player poses and M2 shuttle coordinates. It fits
local before/after velocities, tests direction/velocity changes, checks proximity
to both wrists normalized by bounding-box diagonal, and suppresses nearby duplicate
candidates. Short bounded gaps can be interpolated; long gaps and incomplete
boundary windows are not classified. Possible serves use a rest-to-motion branch.
Strong trajectory evidence without wrists can produce a hit with an unknown hitter;
known wrists far from the shuttle reject a candidate. Identity alternation is only
a review flag, not a forced correction.

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

FP16 can slightly change numerical predictions; use `gpu_half = False` for FP32.

The core video path streams frames; Module 2 retains coordinate records in memory.
Long videos, occlusions and camera cuts can reduce model/tracker accuracy.
Third-party code and weights retain their own licenses.


## Module 5: BST BadmintonDB, six classes

Place `bst_CG_AP_JnB_bone.pt` in `weights/`:
[download the BadmintonDB checkpoint](https://drive.google.com/file/d/1KXY62Q-5kSkgqUiJY88DApdsI2qDqZC5/view).
The external BST checkout is `external/BST-Badminton-Stroke-type-Transformer`.
Install `positional-encodings` and `torchinfo` in the pipeline environment.
The checkpoint has 18 outputs and sequence length 72; incompatible weights fail
strict loading. Inputs are both players' 17 joints + 19 bones, shuttle pixels
normalized by image size, and ankle positions projected through M3's homography.
Top/Bottom slots are ordered by court position, independent of persistent IDs.

After softmax, sum native probabilities (both sides) into these groups:

| ID | Output | Native types |
|---|---|---|
| 0 | Smash | Smash, Net-Kill |
| 1 | Clear | Clear, Net-Lift |
| 2 | Drop | Dropshot |
| 3 | Drive | Drive |
| 4 | Net | Net-Shot, Block |
| 5 | Serve | Serve |

There is no Unknown class. Threshold `0.2` flags low confidence but retains the
winning label; the overlay marks it yellow with `LOW CONF`. Confidence is the
sum of the winning group's probabilities, not a calibrated accuracy estimate.
`shot_type_raw`, `raw_class_id` and `raw_confidence` preserve native 18-way argmax;
its group can differ from the grouped winner. Hitter identity/side comes from
M4 and the court position, independently of native Top/Bottom predictions.

Run all five modules (also the default), writing `shots.mp4` automatically:

```powershell
.\.venv-setup\Scripts\python.exe main.py run data/match178_000.mp4 --modules 1 2 3 4 5 --output outputs/baddb_run
```

Replay only M5 using an existing complete M1-M4 run, then render its labels:

```powershell
.\.venv-setup\Scripts\python.exe main.py classify-shots outputs/run3 --output outputs/baddb_replay
.\.venv-setup\Scripts\python.exe main.py annotate-shots outputs/baddb_replay --output outputs/baddb_replay/shots.mp4
```

Choose a new output path each run. Standalone replay references the source video
through `metadata.json`; it does not copy M1-M4 artifacts. `shot_summary.json`
records the taxonomy, threshold, timing, low-confidence count and skipped shots.
Label display runs from a hit to before the next hit; its final/rally-tail interval
is capped at 72 frames. Model input is separate: previous hit to next hit plus
0.25 seconds, limited to 1.5 seconds before and 1.5 + 0.25 seconds after the target.
Missing neighbors use a 0.5-second fallback on that side. Context is clipped to
the video and known neighboring rally IDs; absent rally IDs are flagged unverified.
This is an experimental adaptation of the upstream ShuttleSet window, using the
same BadmintonDB weight; it has not been validated against BadmintonDB annotations.
JSON records `input_start_frame`, `input_end_frame` (inclusive) and
`input_sample_count` (after sampling, before padding). These input windows may
overlap; `start_frame`/`end_frame` still control non-overlapping video labels.
`bst_min_seg_frames=15` flags short input windows; they still run with padding
and `short_segment`. M5 accepts the two confirmed M1 identities (IDs 1 and 2)
outside court bounds, preserving their projected positions without clipping and
marking `outside_court_player_positions`. It rejects missing/duplicate identities,
invalid poses, missing ankles or invalid homography; an entirely unusable segment
is still reported as skipped. Missing timestamps remain zero-filled; long segments
use upstream temporal sampling. Two-player failures zero pose, position and shuttle.
Summary metadata records the policy and counts of short/outside-court segments.
The relaxed court filter differs from training preprocessing and needs accuracy
validation on labeled footage; padding adds no new motion evidence.

Validate hit boundaries and six-class accuracy on labeled footage before treating
confidence as evidence of accuracy. Grouping alone does not calibrate the model.
