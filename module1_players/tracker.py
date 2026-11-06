"""M1: persistent YOLO11/ByteTrack identities with top-down RTMPose-L."""

from itertools import product

import cv2
import numpy as np
import torch

from config import Settings, require_file
from schemas import Player


class RTMPose:
    def __init__(self, settings: Settings):
        config = require_file(settings.rtmpose_config, "rtmpose_config")
        weights = require_file(settings.rtmpose_weights, "rtmpose_weights")
        try:
            from mmpose.apis import inference_topdown, init_model
            from mmpose.utils import register_all_modules
        except ImportError as exc:
            raise RuntimeError(
                "Install the MMPose stack described in README.md for Module 1"
            ) from exc
        register_all_modules()
        device = torch.device(settings.device)
        if device.type == "cuda":
            device = torch.device("cuda", device.index or 0)
            torch.cuda.set_device(device)
            torch.backends.cudnn.benchmark = True
        self.device = str(device)
        self.half = settings.gpu_half and self.device.startswith("cuda")
        self.model = init_model(str(config), str(weights), device=self.device)
        self.model.eval()
        self.inference = inference_topdown

    def predict(self, frame: np.ndarray, boxes: np.ndarray) -> list[list]:
        if not len(boxes):
            return []  # MMPose otherwise treats an empty list as a full-image person.
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", enabled=self.half, dtype=torch.float16
        ):
            samples = self.inference(self.model, frame, bboxes=boxes, bbox_format="xyxy")
        poses = []
        for sample in samples:
            instance = sample.pred_instances
            xy = np.asarray(instance.keypoints[0])
            scores = np.asarray(instance.keypoint_scores[0])
            if xy.shape != (17, 2) or scores.shape != (17,):
                raise ValueError("RTMPose must return 17 COCO body joints per player")
            poses.append(np.column_stack((xy, scores)).tolist())
        return poses


def foot_point(box, keypoints, threshold: float):
    """Use visible COCO ankles (15,16), falling back to the box bottom center."""
    ankles = [
        keypoints[i][:2]
        for i in (15, 16)
        if keypoints[i][2] >= threshold and np.isfinite(keypoints[i]).all()
    ]
    if ankles:
        return tuple(float(v) for v in np.mean(ankles, axis=0))
    return (float((box[0] + box[2]) / 2), float(box[3]))


class PlayerTracker:
    def __init__(self, settings: Settings, detector=None, pose=None):
        self.settings = settings
        device = torch.device(settings.device)
        if device.type == "cuda":
            device = torch.device("cuda", device.index or 0)
            torch.cuda.set_device(device)
            torch.backends.cudnn.benchmark = True
        self.device = str(device)
        from ultralytics.cfg import DEFAULT_CFG_DICT

        half = settings.gpu_half and device.type == "cuda"
        # Support current and older Ultralytics precision arguments.
        self.precision = (
            {"quantize": 16 if half else 32}
            if "quantize" in DEFAULT_CFG_DICT else {"half": half}
        )
        # Load pose first so missing local assets fail before YOLO downloads anything.
        self.pose = pose if pose is not None else RTMPose(settings)
        if detector is None:
            from ultralytics import YOLO

            detector = YOLO(settings.player_weights)
        self.detector = detector
        self.reset()

    def reset(self):
        """Reset ByteTrack when this instance is reused for another video."""
        self.identities = {}
        predictor = getattr(self.detector, "predictor", None)
        for tracker in getattr(predictor, "trackers", []):
            tracker.reset()

    def _assign_identities(self, candidates, scale):
        """Keep just two IDs, using ByteTrack continuity and nearest last position."""
        if not self.identities:
            if len(candidates) < 2:
                return []
            chosen = sorted(candidates, key=lambda c: c['player'].confidence, reverse=True)[:2]
            # This ordering is used once; never renumber the players on later frames.
            chosen.sort(key=lambda c: c['point'][1])
            assignments = list(zip((1, 2), chosen))
        else:
            def cost(identity, index):
                if index is None:
                    return 1.0
                candidate, previous = candidates[index], self.identities[identity]
                distance = float(np.linalg.norm(candidate['point'] - previous['point'])) / scale
                if distance > .3:
                    return float('inf')
                same_track = candidate['raw_id'] is not None and candidate['raw_id'] == previous['raw_id']
                return distance + (0.0 if same_track else .4)

            # Try the few pairings together so two detections cannot claim one ID.
            options = [None, *range(len(candidates))]
            pairs = (pair for pair in product(options, repeat=2)
                     if pair[0] is None or pair[0] != pair[1])
            best = min(pairs, key=lambda pair: cost(1, pair[0]) + cost(2, pair[1]))
            assignments = [(identity, candidates[index])
                           for identity, index in zip((1, 2), best) if index is not None]
        players = []
        for identity, candidate in assignments:
            self.identities[identity] = dict(point=candidate['point'], raw_id=candidate['raw_id'])
            candidate['player'].track_id = identity
            players.append(candidate['player'])
        return sorted(players, key=lambda p: p.track_id)

    def process(self, frame: np.ndarray, court=None) -> list[Player]:
        result = self.detector.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.settings.player_confidence,
            device=self.device,
            **self.precision,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id
        ids = ids.cpu().numpy().tolist() if ids is not None else [None] * len(boxes)
        # Use detected image-space court points; no guessed metre correspondence is needed.
        points = np.asarray(court.keypoints if court is not None else [], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            return []
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] >= self.settings.court_keypoint_confidence)
        valid &= np.any(points[:, :2] != 0, axis=1)
        if valid.sum() < 4:
            return []  # Never fall back to tracking the crowd when the court disappears.
        hull = cv2.convexHull(points[valid, :2])
        if cv2.contourArea(hull) < .02 * frame.shape[0] * frame.shape[1]:
            return []
        scale = float(np.linalg.norm(np.ptp(hull[:, 0, :], axis=0)))
        # Allow a small border tolerance once the two players are established.
        margin = .04 * scale if self.identities else 0.0
        keep = [i for i, box in enumerate(boxes)
                if cv2.pointPolygonTest(hull, (float((box[0]+box[2])/2), float(box[3])), True) >= -margin]
        boxes, scores, ids = boxes[keep], scores[keep], [ids[i] for i in keep]
        poses = self.pose.predict(frame, boxes)
        if len(poses) != len(boxes):
            raise RuntimeError("RTMPose returned a different number of poses than player boxes")
        candidates = []
        for box, score, track_id, pose in zip(boxes, scores, ids, poses):
            keypoints = np.asarray(pose, dtype=float)
            if keypoints.shape != (17, 3) or not np.isfinite(keypoints).all():
                raise ValueError("Invalid player pose; expected finite (17, 3) keypoints")
            foot = foot_point(box, keypoints, self.settings.pose_confidence)
            if cv2.pointPolygonTest(hull, foot, True) < -margin:
                continue
            player = Player(
                    track_id=int(track_id) if track_id is not None else None,
                    bbox_xyxy=tuple(float(v) for v in box),
                    confidence=float(score),
                    keypoints=[tuple(point) for point in keypoints.tolist()],
                    foot_pixel=foot,
                )
            candidates.append(dict(player=player, point=np.asarray(foot), raw_id=track_id))
        return self._assign_identities(candidates, scale)
