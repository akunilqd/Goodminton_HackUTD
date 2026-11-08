"""M3: one-shot CourtKeyNet calibration for a fixed, full-court camera.

Architecture adapted from https://github.com/adithyanraj03/CourtKeyNet.
Upstream model blobs: octave 68bd4c66, polar e0c9e430, qcm eadfd7d3,
courtkeynet e660f4c0. Keep architecture/state-dict names checkpoint-compatible.

MIT License

Copyright (c) 2025 Adithya N Raj

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import DEFAULT_COURT_WEIGHTS, Settings, require_file
from schemas import Court
from .geometry import COURT_CORNERS_M, fit_homography

"""
Octave Feature Extractor
Manuscript: Section 3.2, Algorithm 1, Equations (1)-(5)
"""


def conv_bn_silu(c_in, c_out, k=3, s=1, p=1):
    """Standard Conv-BN-SiLU block"""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, p, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(inplace=True)
    )


class CourtSpecificShapeKernels(nn.Module):
    """
    Manuscript: Section 3.2.6, Equations (11)-(18)
    Detects horizontal lines, vertical lines, and corners
    """
    def __init__(self, c_in, c_out=64):
        super().__init__()
        # Horizontal line detector (1Ãƒâ€”k kernel)
        self.h_conv = nn.Conv2d(c_in, c_out//4, kernel_size=(1, 5),
                                padding=(0, 2), bias=False)
        # Vertical line detector (kÃƒâ€”1 kernel)
        self.v_conv = nn.Conv2d(c_in, c_out//4, kernel_size=(5, 1),
                                padding=(2, 0), bias=False)
        # Corner detectors (L-shaped patterns)
        self.corner = nn.Conv2d(c_in, c_out//2, kernel_size=3,
                                padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.h_conv(x)
        v = self.v_conv(x)
        c = self.corner(x)
        out = torch.cat([h, v, c], dim=1)
        return self.act(self.bn(out))


class FractalResidualBlock(nn.Module):
    """
    Manuscript: Section 3.2.1, Equation (6)
    Simplified fractal block (recursive structure omitted for stability)
    """
    def __init__(self, c):
        super().__init__()
        self.conv1 = conv_bn_silu(c, c, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class FourierFeatureEncoder(nn.Module):
    """
    Manuscript: Section 3.2.2, Equation (7)
    FFT Ã¢â€ â€™ Learned Frequency Filter Ã¢â€ â€™ IFFT
    """
    def __init__(self, c):
        super().__init__()
        # Learnable complex filter (real and imaginary parts)
        self.filter_re = nn.Parameter(torch.randn(1, c, 1, 1) * 0.02)
        self.filter_im = nn.Parameter(torch.randn(1, c, 1, 1) * 0.02)
        self.proj = nn.Conv2d(c * 2, c, 1)

    def forward(self, x):
        # Convert to float32 for FFT stability
        x_f32 = x.float()

        # Forward FFT
        X = torch.fft.rfft2(x_f32, norm="ortho")

        # Apply learned complex filter (broadcast across spatial frequencies)
        filt = torch.complex(self.filter_re, self.filter_im)
        X_filt = X * filt

        # Inverse FFT
        x_freq = torch.fft.irfft2(X_filt, s=x.shape[-2:], norm="ortho")

        # Concatenate original + frequency-filtered features
        out = torch.cat([x, x_freq.to(x.dtype)], dim=1)
        return self.proj(out)


class NonLocalSelfSimilarity(nn.Module):
    """
    Manuscript: Section 3.2.4, Equation (9)
    Non-local attention for long-range dependencies
    Memory-efficient version with spatial downsampling
    """
    def __init__(self, c, reduction=2, spatial_reduction=8):
        super().__init__()
        c_red = max(c // reduction, 32)
        self.spatial_reduction = spatial_reduction
        self.q = nn.Conv2d(c, c_red, 1)
        self.k = nn.Conv2d(c, c_red, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.out = nn.Conv2d(c, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # Spatially downsample for memory efficiency
        if self.spatial_reduction > 1:
            x_down = F.avg_pool2d(x, self.spatial_reduction)
        else:
            x_down = x

        _, _, H_d, W_d = x_down.shape

        q = self.q(x_down).flatten(2).transpose(1, 2)  # (B, H'd*W'd, C')
        k = self.k(x_down).flatten(2)                   # (B, C', H'd*W'd)
        v = self.v(x_down).flatten(2).transpose(1, 2)   # (B, H'd*W'd, C)

        # Scaled dot-product attention
        attn = torch.softmax(q @ k / (q.shape[-1] ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, C, H_d, W_d)

        # Upsample back to original resolution
        if self.spatial_reduction > 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return x + self.gamma * self.out(out)


class CourtFeatureAggregator(nn.Module):
    """
    Manuscript: Section 3.2.5, Equation (10)
    Multi-orientation filters for view-invariant features
    """
    def __init__(self, c, num_orientations=8):
        super().__init__()
        # Depthwise separable convs with different dilations (orientation proxy)
        self.convs = nn.ModuleList([
            nn.Conv2d(c, c, 3, padding=i+1, dilation=i+1, groups=c, bias=False)
            for i in range(num_orientations)
        ])
        self.fuse = nn.Conv2d(c * num_orientations, c, 1)

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]
        return self.fuse(torch.cat(feats, dim=1))


class OctaveFeatureExtractor(nn.Module):
    """
    Manuscript: Section 3.2, Algorithm 1
    Multi-frequency feature extraction with high/mid/low paths
    """
    def __init__(self, c_band=64):
        super().__init__()

        # Stem network
        self.stem = nn.Sequential(
            conv_bn_silu(3, c_band, 3, 2, 1),
            conv_bn_silu(c_band, c_band, 3, 1, 1)
        )

        # High-frequency path (fine details)
        self.high_path = nn.Sequential(
            CourtSpecificShapeKernels(c_band, c_band),
            FractalResidualBlock(c_band)
        )

        # Mid-frequency path (structural patterns)
        self.mid_path = nn.Sequential(
            CourtSpecificShapeKernels(c_band, c_band),
            FractalResidualBlock(c_band),
            NonLocalSelfSimilarity(c_band)
        )

        # Low-frequency path (global context)
        self.low_path = nn.Sequential(
            CourtFeatureAggregator(c_band),
            FourierFeatureEncoder(c_band)
        )

        # Fusion layer
        self.fuse = nn.Conv2d(c_band * 3, c_band * 2, 1)

    def forward(self, x):
        # Stem
        x = self.stem(x)  # (B, C, H/2, W/2)

        # Split into frequency bands
        f_high = x
        f_mid = F.avg_pool2d(x, 2)  # (B, C, H/4, W/4)
        f_low = F.avg_pool2d(x, 4)  # (B, C, H/8, W/8)

        # Process each frequency band
        f_high = self.high_path(f_high)
        f_mid = self.mid_path(f_mid)
        f_low = self.low_path(f_low)

        # Upsample to match high-frequency resolution
        f_mid = F.interpolate(f_mid, size=f_high.shape[-2:],
                             mode='bilinear', align_corners=False)
        f_low = F.interpolate(f_low, size=f_high.shape[-2:],
                             mode='bilinear', align_corners=False)

        # Fuse all frequencies
        fused = self.fuse(torch.cat([f_high, f_mid, f_low], dim=1))
        return fused


"""
Polar Transform Attention
Manuscript: Section 3.3, Equations (19)-(25)
"""


class PolarTransformAttention(nn.Module):
    """
    Converts features to polar coordinates, processes, then generates attention
    """
    def __init__(self, c, r_bins=64, theta_bins=128):
        super().__init__()
        self.r_bins = r_bins
        self.theta_bins = theta_bins

        # Polar domain convolution
        self.polar_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c)
        )

        # Attention generation
        self.attn_head = nn.Sequential(
            nn.Conv2d(c, c // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c // 4, 1, 1),
            nn.Sigmoid()
        )

    def cartesian_to_polar_grid(self, H, W, device):
        """
        Create sampling grid for Cartesian Ã¢â€ â€™ Polar transformation
        Returns grid in range [-1, 1] for F.grid_sample
        """
        # Polar coordinates
        theta = torch.linspace(0, 2*math.pi, self.theta_bins, device=device)
        r = torch.linspace(0, 1, self.r_bins, device=device)

        # Meshgrid
        R, THETA = torch.meshgrid(r, theta, indexing='ij')

        # Convert to Cartesian (centered at image center)
        X = R * torch.cos(THETA)  # [-1, 1]
        Y = R * torch.sin(THETA)  # [-1, 1]

        # Stack to grid shape (r_bins, theta_bins, 2)
        grid = torch.stack([X, Y], dim=-1)
        return grid

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # 1. Cartesian Ã¢â€ â€™ Polar transformation
        polar_grid = self.cartesian_to_polar_grid(H, W, device)
        polar_grid = polar_grid.unsqueeze(0).expand(B, -1, -1, -1)

        x_polar = F.grid_sample(x, polar_grid, align_corners=False)

        # 2. Process in polar domain
        x_polar = self.polar_conv(x_polar)

        # 3. Interpolate back to Cartesian size
        x_polar_cart = F.interpolate(x_polar, size=(H, W),
                                     mode='bilinear', align_corners=False)

        # 4. Generate attention map
        attn = self.attn_head(x_polar_cart)

        # 5. Apply attention
        return x * attn


"""
Quadrilateral Constraint Module
Manuscript: Section 3.5, Equations (30)-(43)
"""


class QuadrilateralConstraintModule(nn.Module):
    """
    Encodes geometric properties of quadrilateral into feature space
    """
    def __init__(self, feature_dim, hidden_dims=[64, 128]):
        super().__init__()

        # MLP: 10 Ã¢â€ â€™ 64 Ã¢â€ â€™ 128 Ã¢â€ â€™ feature_dim
        layers = []
        dims = [10] + list(hidden_dims) + [feature_dim]

        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(inplace=True)
            ])

        self.mlp = nn.Sequential(*layers[:-1])  # Remove last ReLU

    def compute_geometric_features(self, kpts):
        """
        Compute 10D geometry vector: [d0,d1,d2,d3, d13,d24, ÃŽÂ¸0,ÃŽÂ¸1,ÃŽÂ¸2,ÃŽÂ¸3]
        kpts: (B, 4, 2) normalized coordinates
        """
        B = kpts.shape[0]

        # Edge lengths (4 sides)
        edges = []
        for i in range(4):
            p1 = kpts[:, i]
            p2 = kpts[:, (i+1) % 4]
            dist = torch.norm(p1 - p2, dim=1, keepdim=True)
            edges.append(dist)
        edges = torch.cat(edges, dim=1)  # (B, 4)

        # Diagonal lengths
        d13 = torch.norm(kpts[:,0] - kpts[:,2], dim=1, keepdim=True)
        d24 = torch.norm(kpts[:,1] - kpts[:,3], dim=1, keepdim=True)

        ## Internal angles (using dot product)


        # Internal angles (using cosine directly for numerical stability)
        # Using cos(angle) instead of angle avoids acos gradient issues

        angles = []
        for i in range(4):
            prev = kpts[:, (i-1) % 4]
            curr = kpts[:, i]
            next = kpts[:, (i+1) % 4]

            v1 = prev - curr
            v2 = next - curr

            cos_angle = (v1 * v2).sum(dim=1) / (
                torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-8
            )
            cos_angle = torch.clamp(cos_angle, -1, 1)

            angle = torch.acos(cos_angle).unsqueeze(1)
            angles.append(angle)

            # Use cosine directly (same info, stable gradients)
            #angles.append(cos_angle.unsqueeze(1))


        angles = torch.cat(angles, dim=1)  # (B, 4)

        # Concatenate: (B, 10)
        geom = torch.cat([edges, d13, d24, angles], dim=1)
        return geom

    def forward(self, kpt_features, kpts_init):
        """
        kpt_features: (B, 4, C) - per-keypoint features
        kpts_init: (B, 4, 2) - initial keypoint coordinates
        Returns: (B, 4, C) - enhanced features
        """
        # Compute geometric constraints
        geom_vec = self.compute_geometric_features(kpts_init)  # (B, 10)

        # Encode to feature space
        geom_emb = self.mlp(geom_vec)  # (B, C)

        # Broadcast and add to each keypoint feature
        geom_emb = geom_emb.unsqueeze(1)  # (B, 1, C)

        return kpt_features + geom_emb


"""
CourtKeyNet Main Architecture
Manuscript: Section 3, Figure 2
"""


class CourtKeyNet(nn.Module):
    """
    Complete CourtKeyNet architecture for 4-corner court detection
    """
    def __init__(self, config):
        super().__init__()

        c_band = config['model']['ofe']['channels_per_band']
        c_feat = config['model']['feature_dim']
        num_kpt = config['model']['num_keypoints']

        # 1. Octave Feature Extractor (Backbone)
        self.backbone = OctaveFeatureExtractor(c_band=c_band)

        # 2. Polar Transform Attention
        self.pta = PolarTransformAttention(
            c=c_band * 2,  # OFE outputs 2*c_band
            r_bins=config['model']['pta']['radial_bins'],
            theta_bins=config['model']['pta']['angular_bins']
        )

        # 3. Neck (feature projection)
        self.neck = nn.Sequential(
            nn.Conv2d(c_band * 2, c_feat, 1, bias=False),
            nn.BatchNorm2d(c_feat),
            nn.SiLU(inplace=True)
        )

        # 4. Heatmap Head (coarse localization)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c_feat, c_feat, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_feat),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_feat, num_kpt, 1)
        )

        # 5. Keypoint Feature Extractor
        self.kpt_proj = nn.Linear(c_feat, c_feat)

        # 6. Transformer Encoder (contextual refinement)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c_feat,
            nhead=config['model']['transformer']['num_heads'],
            dim_feedforward=config['model']['transformer']['dim_feedforward'],
            dropout=config['model']['transformer']['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['model']['transformer']['num_layers']
        )

        # 7. Quadrilateral Constraint Module
        self.qcm = QuadrilateralConstraintModule(
            feature_dim=c_feat,
            hidden_dims=config['model']['qcm']['hidden_dims']
        )

        # 8. Offset Regression Head (bounded with tanh to prevent exploding predictions)
        self.offset_head = nn.Sequential(
            nn.Linear(c_feat, c_feat // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c_feat // 2, 2),
            nn.Tanh()  # Bound to [-1, 1]
        )

        # Max offset magnitude (prevents large jumps from initial keypoints)
        self.max_offset = 0.1  # Max 10% of image size

    def soft_argmax_2d(self, heatmaps):
        """
        Differentiable keypoint extraction from heatmaps
        heatmaps: (B, K, H, W)
        Returns: (B, K, 2) normalized coordinates [0, 1]
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device

        # Flatten and softmax
        hm_flat = heatmaps.view(B, K, -1)
        probs = F.softmax(hm_flat, dim=-1)

        # Create coordinate grids
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # (HW, 2)

        # Weighted sum (expectation)
        coords = torch.matmul(probs, grid)  # (B, K, 2)
        return coords

    def forward(self, x):
        """
        x: (B, 3, H, W) input image
        Returns dict with:
          - heatmaps: (B, 4, H', W')
          - kpts_init: (B, 4, 2) initial coordinates
          - kpts_refined: (B, 4, 2) final refined coordinates
        """
        # 1. Feature extraction
        feat = self.backbone(x)        # (B, 128, H/2, W/2)
        feat = self.pta(feat)          # (B, 128, H/2, W/2)
        feat = self.neck(feat)         # (B, C, H/2, W/2)

        # 2. Heatmap prediction
        heatmaps = self.heatmap_head(feat)  # (B, 4, H/2, W/2)

        # 3. Initial keypoint extraction (soft-argmax)
        kpts_init = self.soft_argmax_2d(heatmaps)  # (B, 4, 2)

        # 4. Sample features at keypoint locations
        grid = kpts_init.view(x.size(0), 4, 1, 2) * 2 - 1  # Ã¢â€ â€™ [-1, 1]
        kpt_feats = F.grid_sample(
            feat, grid, align_corners=False
        ).squeeze(-1).permute(0, 2, 1)  # (B, 4, C)

        # 5. Project and refine with transformer
        kpt_feats = self.kpt_proj(kpt_feats)
        kpt_feats = self.transformer(kpt_feats)

        # 6. Apply geometric constraints
        kpt_feats = self.qcm(kpt_feats, kpts_init)

        # 7. Predict refinement offsets (bounded)
        offsets = self.offset_head(kpt_feats) * self.max_offset  # (B, 4, 2), scaled to [-0.1, 0.1]

        # 8. Final refined keypoints
        kpts_refined = torch.clamp(kpts_init + offsets, 0, 1)

        return {
            'heatmaps': heatmaps,
            'kpts_init': kpts_init,
            'kpts_refined': kpts_refined,
            'offsets': offsets
        }

# Configuration shipped with the pinned fine-tuned checkpoint.
_MODEL_CONFIG = {"model": {
    "ofe": {"channels_per_band": 64}, "feature_dim": 128, "num_keypoints": 4,
    "pta": {"radial_bins": 64, "angular_bins": 128},
    "transformer": {"num_heads": 4, "dim_feedforward": 512, "dropout": 0.1, "num_layers": 2},
    "qcm": {"hidden_dims": [64, 128]},
}}
WEIGHTS_REVISION = "8254e285ed0ed496a96fa4632111c80ffb645998"


class CourtDetector:
    """Infer once per reset; subsequent frames reuse geometry, never the network."""

    def __init__(self, settings: Settings, model=None):
        self.settings = settings
        self.init_frame = settings.court_init_frame
        self.device = str(torch.device(settings.device))
        if model is None:
            from safetensors.torch import load_file
            weights = settings.court_weights
            if weights is None and DEFAULT_COURT_WEIGHTS.is_file():
                weights = DEFAULT_COURT_WEIGHTS
            if weights is None:
                from huggingface_hub import hf_hub_download
                download_args = dict(
                    repo_id="Cracked-ANJ/CourtKeyNet",
                    filename="finetuned/courtkeynet_finetuned.safetensors",
                    revision=WEIGHTS_REVISION,
                    cache_dir=str(Path(tempfile.gettempdir()) / "hackutd-courtkeynet-hf"),
                )
                from huggingface_hub.utils import LocalEntryNotFoundError
                try:
                    weights = hf_hub_download(**download_args, local_files_only=True)
                except LocalEntryNotFoundError:
                    weights = hf_hub_download(**download_args)
            require_file(Path(weights), "court_weights")
            model = CourtKeyNet(_MODEL_CONFIG)
            model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        # One-time calibration prioritizes numerical parity over mixed precision.
        self.model = model.to(self.device).float().eval()
        self.reset()

    def reset(self):
        self.court = None
        self.image_size = None
        self.inference_count = 0

    def process(self, frame: np.ndarray) -> Court:
        size = (frame.shape[1], frame.shape[0])
        if self.court is not None:
            if size != self.image_size:
                raise ValueError("Camera resolution changed; reset court calibration")
            return self.court
        width, height = size
        rgb = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.
        with torch.inference_mode():
            outputs = self.model(tensor.to(self.device))
            normalized = outputs["kpts_refined"][0].float().cpu().numpy()
            probs = outputs["heatmaps"][0].float().flatten(1).softmax(-1)
            peaks = probs.amax(-1)
            peak_score = ((peaks - .05) / .35).clamp(0, 1)
            entropy_score = 1 + (probs * probs.clamp_min(1e-10).log()).sum(-1) / math.log(probs.shape[-1])
            # Upstream GUI weights: peak .4, valid geometry .4, entropy .2.
            # Shared court score, as in upstream GUI; repeated for the four corners.
            # Geometry contributes only for the hard-valid polygon checked below.
            # These are heuristic quality scores, NOT calibrated probabilities.
            quality = (.4 * peak_score.mean() + .4 + .2 * entropy_score.mean()).item()
            scores = np.full(4, quality, dtype=np.float32)
        self.inference_count += 1
        if normalized.shape != (4, 2) or scores.shape != (4,):
            raise ValueError("CourtKeyNet must return four corners in TL, TR, BR, BL order")
        if not np.isfinite(normalized).all() or not np.isfinite(scores).all():
            raise ValueError("CourtKeyNet returned non-finite predictions")
        xy = normalized * np.array([width, height])
        polygon = xy.astype(np.float32)
        center = normalized.mean(axis=0)
        order_ok = (
            normalized[0, 0] < center[0] and normalized[0, 1] < center[1]
            and normalized[1, 0] > center[0] and normalized[1, 1] < center[1]
            and normalized[2, 0] > center[0] and normalized[2, 1] > center[1]
            and normalized[3, 0] < center[0] and normalized[3, 1] > center[1]
        )
        area = cv2.contourArea(polygon) / (width * height)
        edges = np.linalg.norm(np.roll(normalized, -1, axis=0) - normalized, axis=1)
        aspect = (edges[0] + edges[2]) / max(float(edges[1] + edges[3]), 1e-8)
        if (not cv2.isContourConvex(polygon) or not .02 < area < .95
                or not .3 < aspect < 5. or not order_ok or np.any(normalized <= 0) or np.any(normalized >= 1)):
            raise ValueError("Invalid/full-court corners; select another initialization frame or manual calibration")
        if np.any(scores < self.settings.court_keypoint_confidence):
            raise ValueError(
                f"Low CourtKeyNet corner scores {scores.tolist()}; "
                "inspect another initialization frame or use manual calibration"
            )
        matrix, count, error = fit_homography(
            xy, COURT_CORNERS_M, self.settings.court_ransac_threshold_m,
            self.settings.court_max_error_m,
        )
        self.court = Court(
            keypoints=[tuple(p) for p in np.column_stack((xy, scores)).tolist()],
            homography=matrix, inlier_count=count, reprojection_error_m=error,
            status="fixed_courtkeynet",
        )
        # Four-point fit residual is NOT an independent accuracy measurement.
        self.image_size = size
        return self.court


class ManualCourt:
    """Explicit fixed-camera calibration; never silently reused at another resolution."""

    def __init__(self, path: Path, settings: Settings):
        self.settings = settings
        with require_file(path, "court_calibration").open(encoding="utf-8") as stream:
            data = json.load(stream)
        self.image_size = tuple(data["image_size"])
        if len(self.image_size) != 2 or any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in self.image_size
        ):
            raise ValueError("Calibration image_size must be [width,height] in positive pixels")
        points = np.asarray(data["image_points"], dtype=float)
        matrix, count, error = fit_homography(
            points,
            data["court_points_m"],
            settings.court_ransac_threshold_m,
            settings.court_max_error_m,
        )
        self.court = Court(
            keypoints=[(float(x), float(y), 1.0) for x, y in points],
            homography=matrix,
            inlier_count=count,
            reprojection_error_m=error,
            status="manual_calibration",
        )

    def process(self, frame: np.ndarray) -> Court:
        if (frame.shape[1], frame.shape[0]) != self.image_size:
            raise ValueError("Video resolution differs from manual calibration image_size")
        return self.court
