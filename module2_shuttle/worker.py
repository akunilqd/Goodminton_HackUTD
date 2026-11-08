"""Run the upstream shuttle models in isolation and return their results on stdout."""

import argparse
import json
import math
import sys
from contextlib import redirect_stdout
from itertools import islice
from pathlib import Path


def infer(args):
    # The adapter starts this worker from the upstream TrackNetV3 checkout.
    sys.path.insert(0, str(Path.cwd()))
    import torch
    from torch.utils.data import DataLoader, default_collate
    from tqdm import tqdm

    from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
    from predict import predict
    from test import generate_inpaint_mask
    from utils.general import COOR_TH, get_model

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    dtype = torch.float16 if args.half and device.type == "cuda" else torch.float32

    checkpoint = torch.load(args.tracknet, map_location="cpu", weights_only=False)
    seq_len = checkpoint["param_dict"]["seq_len"]
    bg_mode = checkpoint["param_dict"]["bg_mode"]
    tracknet = get_model("TrackNet", seq_len, bg_mode)
    tracknet.load_state_dict(checkpoint["model"])
    del checkpoint
    tracknet = tracknet.to(device=device, dtype=dtype).eval()

    video = Video_IterableDataset(args.video, seq_len=seq_len, sliding_step=seq_len, bg_mode=bg_mode)
    scaler = (video.w_scaler, video.h_scaler)
    raw = {key: [] for key in ("Frame", "X", "Y", "Visibility")}
    try:
        if video.video_len <= 0:
            raise ValueError("TrackNet could not read the input video")
        sequence_count = math.ceil(video.video_len / seq_len)
        # Bound the iterator to avoid probing the upstream empty-tail sequence.
        sequences = islice(video, sequence_count)
        batch_count = math.ceil(sequence_count / args.batch_size)
        with torch.inference_mode():
            for _ in tqdm(range(batch_count), desc="TrackNet"):
                batch = list(islice(sequences, args.batch_size))
                if not batch:
                    break
                indices, frames = default_collate(batch)
                heatmaps = tracknet(frames.to(device=device, dtype=dtype)).cpu()
                positions = predict(indices, y_pred=heatmaps, img_scaler=scaler)
                for key in raw:
                    raw[key].extend(positions[key])
    finally:
        video.cap.release()
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
    with redirect_stdout(sys.stderr):
        predictions = infer(parser.parse_args())
    json.dump(predictions, sys.stdout, allow_nan=False)
