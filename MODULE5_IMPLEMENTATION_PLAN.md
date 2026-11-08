# Module 5: BST implementation plan

Investigation date: 2026-11-07. Scope: implementation planning; no inference code or dependencies changed.

## Recommendation

Reuse the existing M1-M3 frame records. Add a checkpoint-compatible feature adapter and a stroke-window builder, then run BST once per hit. Start with offline classification using the upstream 25-class, 2D, JnB_bone, sequence-length-100 BST_CG_AP example as a reproducible baseline, not an assumed best model. Benchmark alternatives before selecting a production checkpoint. Keep native labels and expose six product categories only through an explicit mapping with Unknown/Other support.

The first dependencies are court calibration and hit-event boundaries. Neither is currently available in the saved run. Manually annotated hit events allow M5 development and evaluation before M4 is implemented.

## Evidence from this workspace

Reviewed SYSTEM_DESIGN.md, schemas.py, config.py, pipeline.py, main module implementations, README.md, requirements.txt, and saved outputs.

| Existing output | Integration | Required adaptation |
| --- | --- | --- |
| M1 `bbox_xyxy`, 17 COCO `keypoints` with x/y/score | Compatible source for 2D pose | Normalize x/y using each player's box; keep confidence separately; derive upstream bone features |
| M1 persistent `track_id` | Useful for attribution | BST expects Top then Bottom slots, not ID order or hitter first; preserve slot-to-track mapping |
| M1 `foot_pixel`, `court_xy_m` | Approximate player ground position | For baseline parity, project each ankle separately and then average; do not reuse the projected image-space midpoint |
| M2 `pixel`, `raw_pixel`, visibility and provenance | Compatible shuttle trajectory | Divide image x/y by source width/height; choose repaired/raw consistently; preserve missingness outside input tensors |
| M3 `homography` and court keypoints | Needed for position-aware BST variants | Supply a verified template or manual fixed-camera calibration; establish origin and axis direction |
| `frame_index`, `timestamp_s`, `metadata.json` | Suitable alignment contract | Keep zero-based indexing, source FPS and dimensions; validate contiguous frames and complete run status |

Measured by parsing existing JSONL, without rerunning models:

| Audit | `outputs/run2` |
| --- | --- |
| Video | 1920 x 1080, 25 FPS, 250 frames / 10 seconds |
| Players | Two in 247 frames; one in 3 frames |
| Pose shape | 17 x 3 for every recorded player |
| Court | `template_required` in all 250 frames; zero valid homographies |
| Shuttle source | 92 `tracknet`, 157 `inpaintnet`, 1 `missing` |

The 157 changed positions are provenance counts, not an accuracy or confidence measurement. `rectified` means final coordinates differ from raw coordinates. Inspect hit neighborhoods for trajectory errors rather than assuming every repaired position is reliable.

`run1` predates the two-player filter and contains 7-9 people per frame. Do not feed that export directly to BST. `run2` is the appropriate starting fixture, but it has no ground-truth shot labels and cannot establish classification accuracy.

M1 currently depends on M3 image-space court points for player filtering even without a homography. It reserves IDs 1 and 2 at initialization and does not renumber them after side changes. Missing detections are omitted. M2 runs TrackNet twice, once raw and once with InpaintNet. The pipeline completes these whole-video passes before processing M1/M3. This is currently an offline pipeline.

## Corrections to the system-design assumptions

1. Use the user-supplied [BST repository](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer), not the different URL in SYSTEM_DESIGN.md.
2. The upstream [inference example](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/blob/main/stroke_classification/main_on_shuttleset/bst_infer.py) uses both players' poses, shuttle positions, player court positions and sequence lengths. It does not consume raw RGB frames or only the hitter's pose.
3. ShuttleSet checkpoints are not six-class heads. The [dataset code](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/blob/main/stroke_classification/preparing_data/shuttleset_dataset.py) defines 25 classes as Unknown + 12 Top + 12 Bottom, and 35 classes as 17 Top + 17 Bottom + Unknown. Unknown has a different index in these two lists. Never derive IDs alphabetically.
4. Weights are linked from the upstream README through Google Drive. Availability, exact checkpoint contents and local compatibility have not yet been tested.
5. The [paper](https://arxiv.org/abs/2502.21085) addresses singles. The current two-player pipeline fits that scope; doubles requires a separate design/training effort.

## Exact feature adapter

Use one versioned preprocessing profile per checkpoint. Record source revision, weight SHA-256, architecture, full ordered labels, sequence length, joint order, bone pairs, normalization flags, window policy and missing-data policy.

For the example `BST_CG_AP`, `JnB_bone`, 2D, 25-class, T=100 configuration:

| Tensor | Shape |
| --- | --- |
| Normalized joints before bones | `[B, 100, 2, 17, 2]` |
| Pose after 19 bones and flattening | `[B, 100, 2, 72]` |
| Shuttle | `[B, 100, 2]` |
| Player position | `[B, 100, 2, 2]` |
| `video_len` | `[B]`, integer effective lengths after temporal selection |
| Logits | `[B, 25]` |

Call order is `model(pose, shuttle, position, video_len)`. Confidence is not a third coordinate channel. J_only instead has a 34-dimensional pose input and requires a matching checkpoint.

Implement from the [upstream preparation code](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/blob/main/stroke_classification/preparing_data/prepare_train_on_shuttleset.py):

- Pose: for ordinary nonzero joints with center alignment, `(joint - box_center) / box_diagonal`. Match upstream treatment of zero coordinates exactly in the parity profile. The helper defaults to `center_align=False`, while the preparation example passes `True`; resolve against the chosen checkpoint, rather than trusting a helper default.
- Bones: use the exact 19 ordered upstream pairs and its coordinate-wise zero handling. The visualization edges in pipeline.py are not the BST bone definition.
- Position: project left and right ankles (COCO indices 15 and 16) independently through H, average in court space, then normalize court boundaries. With far-left origin, matching orientation and full doubles court bounds, this becomes `(x / 6.10, y / 13.40)`. Verify those bounds/orientation against reference examples. Do not apply upstream's 1280x720 resolution conversion to our H, which already maps original-image pixels to metres.
- Player order: reproduce Top/Bottom sorting by normalized court y for the compatibility baseline. Keep persistent IDs separately for analytics and hitter attribution. Reject windows crossing camera cuts or side-switch transitions rather than mixing people.
- Shuttle: `(x / width, y / height)` using image pixels. Do not use `court_projection_m`; an airborne shuttle is not on the court plane.
- Sequence length: port `make_seq_len_same` exactly. It uses stride selection/truncation for long sequences and trailing zero padding for short sequences, not generic linear resampling. Match effective lengths and padding before experimenting with better sampling.
- Missing data: upstream zeros both players' features and the shuttle on failed two-player frames. Reproduce this for baseline parity, retain quality flags separately, and reject all-invalid windows. `video_len` masks trailing padding, not arbitrary internal missing frames. Confidence masks or interpolation change the input distribution and should be evaluated/fine-tuned, not added as new channels to existing weights.

The upstream `model/bst.py` also defines `BST_0` without a position input, while README/training variant names and inference names differ. Select by exact constructor, forward signature and state-dict keys, not by the text “BST-0.” A no-position fallback needs its own compatible weights and evaluation; filling all position tensors with zeros is not an equivalent substitute.

## Hit events and temporal windows

Introduce `HitEvent`: `event_id`, `rally_id`, `frame_index`, `timestamp_s`, nullable `hitter_id`, hit confidence, event source, and adjacent-hit references. Accept manually labeled events first; later consume M4 without changing the M5 API.

The [upstream clip generator](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer/blob/main/ShuttleSet/gen_my_dataset.py) uses the previous hit through the next hit plus approximately 0.25 seconds, capped near 1.5 seconds before and 1.75 seconds after the current hit. Rally-edge fallbacks use approximately 0.5 seconds. Reproduce its FPS-based rounding and end-exclusive clipping semantics, and test conversion from dataset annotations to our zero-based frames.

Consequences:

- T=100 is the network's sampled/padded length, not necessarily 100 raw consecutive frames or a fixed physical duration.
- A faithful final prediction needs future context, potentially about 1.75 seconds plus event-detection and compute delay. A shorter tensor alone does not remove this wait.
- First/last rally hits, truncated video boundaries, duplicate hits and clips spanning cuts need explicit policies and quality flags.
- Until M4 exists, label rally boundaries and hit frames manually on representative clips. This separates M5 errors from hit-detection errors.
- For future M4, combine trajectory changes with wrist proximity and temporal suppression; evaluate hit precision/recall, timing error and hitter attribution. Do not treat a wrist as the racket contact point or blindly enforce perfect player alternation.

## Label policy

Keep `native_class_id`, native label, native probability vector, predicted side and mapped persistent player ID. Add `shot_type`, aggregated probability, rejection reason and feature-quality fields for consumers.

Initial unambiguous mappings for the merged 25-class model, applied to both Top and Bottom:

| Native type | Product type |
| --- | --- |
| 殺球 | Smash |
| 長球 | Clear |
| 切球 | Drop |
| 平球 | Drive |
| 挑球 | Lift |
| 發短球, 發長球 | Serve |
| 未知球種 | Unknown |
| 放小球, 擋小球, 推球, 撲球, 勾球 | Other until a reviewed product taxonomy specifies their treatment |

The six requested types do not naturally cover net shots, blocks, pushes, net kills and related classes. Do not silently force these into Drop or Drive. Preserve native predictions for future richer analytics. Sum softmax probabilities over mapped classes, including both sides when computing side-independent shot probabilities; retain Other/Unknown mass rather than renormalizing it away. If conditioning on M4 hitter side, report that as a separately validated policy and flag strong disagreements.

If exactly six action labels are required, define how unsupported actions are excluded/rejected and fine-tune a six-class head on consistent labels. Replacing the head with six random outputs is not pretrained six-class inference. Keep a quality/unknown status even with a six-class action head.

## Implementation phases and deliverables

1. **Freeze the baseline.** Pin upstream revision, select and download a matching checkpoint, save a model manifest, verify strict state-dict loading and known-example predictions. Check the current PyTorch 2.1 environment before importing upstream code: its README uses 2.4, and model imports include profiling utilities. Isolate inference-only dependencies; use a separate long-lived BST environment if necessary rather than upgrading the working MMPose stack. Update the M5 section of SYSTEM_DESIGN.md to the verified contract.
2. **Unblock inputs.** Calibrate the fixed camera with verified floor correspondences; inspect overlays and normalized ankle locations. Compute a derived feature cache from run2 plus calibration without rerunning pose/shuttle networks. Record calibration provenance. Annotate hits/rallies and a small set of shot labels for integration fixtures.
3. **Implement compatibility preprocessing.** Add `module5_shots/features.py`, `windows.py`, `labels.py`, `classifier.py`, and `__init__.py`; keep imported model code in an isolated namespace. Store a manifest under `configs/`. Add event/output schemas in schemas.py. Test tensors against upstream functions on identical fixtures before any optimization.
4. **Integrate offline inference.** Add a proposed `main.py classify` subcommand accepting an existing complete run, hit-event file and model manifest. Load BST once; stream or cache features; batch events; write `shots.jsonl` and classification metadata to a separate output directory. Later wire M4/M5 into `main.py run`. Do not rerun M1-M3 for classifier iterations.
5. **Measure and improve accuracy.** Establish held-out results with manual hits, then predicted hits. Fine-tune only if the baseline or product taxonomy requires it. Compare raw versus repaired shuttle input, quality gating, and pose-gap repair on validation data.
6. **Optimize measured bottlenecks.** Benchmark eager FP32 and mixed precision first. Compare matching BST variants/checkpoints and T=30 versus T=100 profiles. Add export/runtime optimization only if it improves measured end-to-end latency without unacceptable accuracy loss.

Suggested `ShotPrediction` fields: event/rally IDs, hit frame/time, actual window bounds, `hitter_id`, predicted side, native class/label, product label, probabilities, accepted/unknown/insufficient-data status, quality reasons, model/preprocessing version, compute duration and finalization time. Preserve parent-run and calibration identifiers in metadata.

## Accuracy strategy

- Build a labeled evaluation set spanning different matches/cameras, near/far players, rare shots, serves, occlusions and missing detections. A small manually labeled smoke set validates plumbing; credible accuracy claims require enough independent events per class.
- Split by match/video, with player-disjoint evaluation where practical. Never randomly split neighboring overlapping stroke windows across train and test. Audit overlap with the checkpoint's training matches when claiming generalization.
- Report macro-F1, per-class precision/recall, confusion matrices and overall accuracy for native and product labels. Also report hitter accuracy, rejection coverage and performance on missing/repaired-input subsets. Report confidence intervals with match-aware resampling when sufficient matches exist.
- Measure M5 with oracle hits separately from the complete pipeline with predicted hits. Include unmatched/missed/spurious events in end-to-end evaluation; classification accuracy on detected hits alone hides M4 failures.
- Fine-tune using features generated by this exact RTMPose-L and qaz812345 TrackNet/InpaintNet pipeline. Upstream preprocessing uses MMPose's `human` configuration and reports the attention-based TrackNet variant, so compatible shapes do not establish matching feature distributions.
- Start with head adaptation, then cautiously unfreeze the backbone with validation-selected learning rates. Evaluate class weighting/balanced sampling for rare strokes, realistic pose noise/missingness, and hit-time jitter. Apply geometric augmentation consistently across modalities and remap sides/joints/labels when necessary.
- Select confidence thresholds and probability calibration on validation data. Track accuracy-versus-coverage; do not claim increased accuracy simply by rejecting most events.
- Maintain singles scope. Track loss, camera cuts and long gaps should yield explicit insufficient-data results rather than confident fabricated labels.

## Inference speed strategy

1. Reuse numerical outputs; decode and run upstream networks once per video. Cache normalized per-frame features and build overlapping hit windows through indexing.
2. Keep one BST instance resident, use evaluation/inference mode, preallocate contiguous buffers where useful, and transfer event batches rather than individual joints/frames. Benchmark batch sizes 1, 8, 16 and 32 against memory and latency; do not inherit upstream's batch=128 blindly.
3. Establish FP32 parity, then test CUDA mixed precision. Keep homography/normalization in FP32 or FP64. Compare logits, labels, macro-F1 and confidence calibration before enabling FP16 by default.
4. Treat ONNX Runtime/TensorRT or compilation as optional measured experiments. Check custom attention, padding masks, numerical behavior and Windows/runtime compatibility. Static T and small batch buckets are initial export candidates. Include startup/build time and warm-up separately.
5. Do not shorten a T=100 model by passing 30 frames: learned positional embeddings and training windows must match. Use matching T=30 weights or retrain/distill a shorter-context model.
6. For full-pipeline speed, remove duplicate TrackNet computation by exporting raw and repaired trajectories from one run, preserving existing alignment/provenance. Consider court refresh intervals only with validated cut/movement detection and a valid cached court region, because M1 uses court points every frame.
7. Avoid concurrent GPU-heavy subprocesses until memory is measured. The README records an RTX 3050 6 GB setup; establish current available hardware and per-stage peak memory before selecting batches or running models together.

Benchmark preprocessing, host/device transfers, BST forward, output serialization, full-video wall time and peak VRAM separately. Use GPU synchronization/timing events, warm-up, repeated representative windows and p50/p95 latency. Report both single-event latency and batched events/second.

Provisional engineering targets, not measured promises: M5 preprocessing + transfers + forward p95 below 30 ms/event at batch 1 on the target GPU; FP16/export macro-F1 degradation no more than 0.5 percentage points versus FP32. Establish a labeled baseline before setting an absolute accuracy target. Report full-pipeline throughput independently; the existing README's older M1/M3 frame benchmark is not evidence of real-time end-to-end operation.

## Optional low-latency mode after the offline baseline

A live system requires a streaming M2/M4 path, not only a fast BST call. Train or fine-tune a fixed-lookahead/causal classifier with the same context available at deployment. For example, evaluate a 200-300 ms provisional lookahead and emit a later revised label once the full reference window is available. Treat this as a separate model/profile and measure its accuracy-latency tradeoff. Do not present truncated-context predictions from the offline checkpoint as equivalent to validated final predictions.

## Acceptance checks

- Strict checkpoint load, exact label ordering and reproducible manifest/hash.
- Golden tensor parity for normal inputs, missing players/shuttle, zero joints, bones, Top/Bottom ordering, ankle projections, long-window sampling and trailing padding.
- Hit-window tests at rally/video boundaries, 25/30/60 FPS, frame-offset conversions, camera cuts and effective-length masking.
- Integration from existing frames + calibration + manual hits to one auditable record per event, with no rerun of M1-M3 and explicit handling of unusable events.
- Batch-size invariance within numerical tolerance and FP32/FP16/export comparisons on representative valid and degraded windows.
- Held-out accuracy/coverage and full timing report, with oracle-hit and predicted-hit results separated.

Open prerequisites: verified camera calibration; selected downloadable checkpoint and associated preprocessing settings; labeled hit/shot evaluation data; product decision on six-class treatment of net/other shots. These do not prevent implementing the offline adapter and native-label baseline first.
