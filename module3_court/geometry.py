"""M3 geometry. Coordinates are metres across (x) and along (y) the court."""

import cv2
import numpy as np

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40


def project_point(homography, point):
    if homography is None or point is None:
        return None
    matrix = np.asarray(homography, dtype=float)
    pixel = np.asarray([*point, 1.0], dtype=float)
    mapped = matrix @ pixel
    if not np.isfinite(mapped).all() or abs(mapped[2]) < 1e-10:
        return None
    return (float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2]))


def on_court(point, margin_m: float = 0.0):
    if point is None:
        return None
    return (
        -margin_m <= point[0] <= COURT_WIDTH_M + margin_m
        and -margin_m <= point[1] <= COURT_LENGTH_M + margin_m
    )


def fit_homography(image_points, court_points, threshold_m=0.15, max_error_m=0.2):
    """Reject degenerate/poor fits; the RANSAC threshold is in destination metres."""
    image_points = np.asarray(image_points, dtype=np.float64)
    court_points = np.asarray(court_points, dtype=np.float64)
    if (
        image_points.ndim != 2
        or image_points.shape[1] != 2
        or image_points.shape != court_points.shape
        or len(image_points) < 4
    ):
        raise ValueError("Homography requires at least four paired 2D points")
    if not np.isfinite(image_points).all() or not np.isfinite(court_points).all():
        raise ValueError("Homography points must be finite")
    for points in (image_points, court_points):
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("Homography points must not be collinear")
    matrix, mask = cv2.findHomography(image_points, court_points, cv2.RANSAC, threshold_m)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        raise ValueError("Could not estimate a valid homography")
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Homography is singular")
    inliers = mask.ravel().astype(bool)
    count = int(inliers.sum())
    if count < 4 or count / len(image_points) < 0.5:
        raise ValueError("Too few consistent court keypoints")
    projected = cv2.perspectiveTransform(image_points[inliers, None, :], matrix)[:, 0, :]
    error = float(np.sqrt(np.mean(np.sum((projected - court_points[inliers]) ** 2, axis=1))))
    if not np.isfinite(error) or error > max_error_m:
        raise ValueError(f"Court reprojection error {error:.3f} m exceeds {max_error_m} m")
    return matrix.tolist(), count, error
