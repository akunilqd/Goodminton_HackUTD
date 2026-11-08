"""Run the upstream shuttle models in isolation and return their results on stdout."""

import argparse
import json
import math
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MEDIAN_SAMPLES = 1800  # Upstream's max_sample_num; decides which frames are sampled.


def median_background(path, frame_count, size, pool, workers):
    """Upstream's concat background, without per-frame seeks or an in-memory frame stack."""
    capture = cv2.VideoCapture(path)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    step = frame_count // MEDIAN_SAMPLES if frame_count > MEDIAN_SAMPLES else 1
    histogram = np.zeros((height, width, 3, 256), dtype=np.uint16)
    bands = [(int(rows[0]), int(rows[-1]) + 1) for rows in np.array_split(np.arange(height), workers)]
    positions = np.arange(width * 3 * max(bottom - top for top, bottom in bands))

    def count(top, bottom, frame):
        cells = histogram[top:bottom].reshape(-1, 256)
        # One frame touches each cell once, so fancy-index increments cannot collide.
        cells[positions[:len(cells)], frame[top:bottom].ravel()] += 1

    samples = 0
    try:
        for index in range(0, frame_count):
            if not capture.grab():
                break
            if index % step == 0:
                frame = capture.retrieve()[1]
                list(pool.map(lambda band: count(*band, frame), bands))
                samples += 1
    finally:
        capture.release()
    if samples == 0:
        raise ValueError("TrackNet could not read the input video")

    def middle(band):
        block = histogram[band[0]:band[1]]
        np.cumsum(block, axis=-1, out=block)
        lower = (block > (samples - 1) // 2).argmax(-1).astype(np.uint16)
        upper = (block > samples // 2).argmax(-1).astype(np.uint16)
        return ((lower + upper) // 2).astype(np.uint8)  # Truncates like upstream's astype.

    median = cv2.cvtColor(np.concatenate(list(pool.map(middle, bands))), cv2.COLOR_BGR2RGB)
    return np.moveaxis(np.asarray(Image.fromarray(median).resize(size)), -1, 0)


def resized_frames(path, size, pool, depth):
    """Yield RGB (3, H, W) frames resized like upstream, preparing `depth` ahead of the GPU."""
    capture = cv2.VideoCapture(path)

    def resize(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.frombuffer("RGB", rgb.shape[1::-1], rgb, "raw", "RGB", 0, 1).resize(size)
        return np.moveaxis(np.asarray(image), -1, 0)

    def read_next():
        success, frame = capture.read()
        return pool.submit(resize, frame) if success else None

    try:
        with ThreadPoolExecutor(1) as decoder:
            pending = deque(decoder.submit(read_next) for _ in range(depth))
            while (job := pending.popleft().result()) is not None:
                pending.append(decoder.submit(read_next))
                yield job.result()
    finally:
        capture.release()


def input_sequences(frames, median, seq_len, count):
    """Non-overlapping upstream windows; the final short window repeats its last frame."""
    for start in range(0, count * seq_len, seq_len):
        window = list(islice(frames, seq_len))
        if not window:
            return
        ids = list(range(start, start + len(window)))
        ids += ids[-1:] * (seq_len - len(ids))
        window += window[-1:] * (seq_len - len(window))
        yield [(0, index) for index in ids], np.concatenate([median, *window])


def infer(args):
    # The adapter starts this worker from the upstream TrackNetV3 checkout.
    sys.path.insert(0, str(Path.cwd()))
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
    from predict import predict
    from test import generate_inpaint_mask
    from utils.general import COOR_TH, HEIGHT, WIDTH, get_model

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    dtype = torch.float16 if args.half and device.type == "cuda" else torch.float32

    checkpoint = torch.load(args.tracknet, map_location="cpu", weights_only=False)
    seq_len = checkpoint["param_dict"]["seq_len"]
    bg_mode = checkpoint["param_dict"]["bg_mode"]
    if bg_mode != "concat":
        raise ValueError(f"Worker supports concat background checkpoints; got {bg_mode!r}")
    tracknet = get_model("TrackNet", seq_len, bg_mode)
    tracknet.load_state_dict(checkpoint["model"])
    del checkpoint
    tracknet = tracknet.to(device=device, dtype=dtype).eval()

    probe = cv2.VideoCapture(args.video)
    frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    if frame_count <= 0:
        raise ValueError("TrackNet could not read the input video")
    size = (WIDTH, HEIGHT)
    # Same value as dividing float64 pixels by 255 and casting, as upstream's loader does.
    normalized = (torch.arange(256, dtype=torch.float64) / 255).to(device=device, dtype=dtype)

    raw = {key: [] for key in ("Frame", "X", "Y", "Visibility")}
    with ThreadPoolExecutor(args.workers) as pool:
        median = median_background(args.video, frame_count, size, pool, args.workers)
        video = Video_IterableDataset(
            args.video, seq_len=seq_len, sliding_step=seq_len, bg_mode=bg_mode, median=median,
        )
        video.cap.release()
        scaler = (video.w_scaler, video.h_scaler)
        sequence_count = math.ceil(frame_count / seq_len)
        frames = resized_frames(args.video, size, pool, seq_len * args.batch_size * 2)
        sequences = input_sequences(frames, median, seq_len, sequence_count)
        batch_count = math.ceil(sequence_count / args.batch_size)
        with torch.inference_mode():
            for _ in tqdm(range(batch_count), desc="TrackNet"):
                batch = list(islice(sequences, args.batch_size))
                if not batch:
                    break
                indices = torch.tensor([ids for ids, _ in batch])
                pixels = torch.from_numpy(np.stack([window for _, window in batch]))
                heatmaps = tracknet(normalized[pixels.to(device).long()]).cpu()
                positions = predict(indices, y_pred=heatmaps, img_scaler=scaler)
                for key in raw:
                    raw[key].extend(positions[key])
        frames.close()
    if not raw["Frame"]:
        raise ValueError("TrackNet returned no frames")
    del tracknet

    checkpoint = torch.load(args.inpaintnet, map_location="cpu", weights_only=False)
    seq_len = checkpoint["param_dict"]["seq_len"]
    inpaintnet = get_model("InpaintNet")
    inpaintnet.load_state_dict(checkpoint["model"])
    del checkpoint
    inpaintnet = inpaintnet.to(device).eval()

    trajectory = {
        **raw,
        "Inpaint_Mask": generate_inpaint_mask(raw, th_h=video.h * 0.05),
        "Img_scaler": scaler,
        "Img_shape": (video.w, video.h),
    }
    windows = Shuttlecock_Trajectory_Dataset(
        seq_len=seq_len, sliding_step=seq_len, data_mode="coordinate",
        pred_dict=trajectory, padding=True,
    )
    repaired = {key: [] for key in raw}
    with torch.inference_mode():
        for indices, coords, mask in tqdm(DataLoader(windows, batch_size=args.batch_size), desc="InpaintNet"):
            coords, mask = coords.float(), mask.float()
            filled = inpaintnet(coords.to(device), mask.to(device)).cpu()
            filled = filled * mask + coords * (1 - mask)
            filled[(filled[:, :, 0] < COOR_TH) & (filled[:, :, 1] < COOR_TH)] = 0.0
            positions = predict(indices, c_pred=filled, img_scaler=scaler)
            for key in repaired:
                repaired[key].extend(positions[key])
    return raw, repaired


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("tracknet")
    parser.add_argument("inpaintnet")
    parser.add_argument("device")
    parser.add_argument("batch_size", type=int)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    with redirect_stdout(sys.stderr):
        predictions = infer(parser.parse_args())
    json.dump(predictions, sys.stdout, allow_nan=False)
