# projects/project3/code/compute_homography.py
# Homography via linear least squares in the "Ah = b" form with 8 unknowns.
# We parameterize H as:
#     H = [[h1 h2 h3],
#          [h4 h5 h6],
#          [h7 h8  1]]
# so the unknown vector is h = [h1,...,h8]^T and H[2,2]=1.

from __future__ import annotations
import numpy as np

def build_Ab(im1_pts: np.ndarray, im2_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the linear system A h = b for homography estimation with H[2,2]=1.

    im1_pts: (N,2) source points (x,y)
    im2_pts: (N,2) destination points (x',y')
    Returns:
        A: (2N, 8)   matrix
        b: (2N,)     vector
    For each correspondence (x,y) -> (x',y'):
        x' = (h1 x + h2 y + h3) / (h7 x + h8 y + 1)
        y' = (h4 x + h5 y + h6) / (h7 x + h8 y + 1)
    Rearranged into linear equations in h1..h8:
        [ x  y  1  0  0  0  -x x'  -y x' ] · h = x'
        [ 0  0  0  x  y  1  -x y'  -y y' ] · h = y'
    """
    im1_pts = np.asarray(im1_pts, dtype=float)
    im2_pts = np.asarray(im2_pts, dtype=float)
    assert im1_pts.shape == im2_pts.shape and im1_pts.ndim == 2 and im1_pts.shape[1] == 2, \
        "im1_pts and im2_pts must both be (N,2)"
    N = im1_pts.shape[0]
    assert N >= 4, "Need at least 4 correspondences"

    x, y   = im1_pts[:, 0], im1_pts[:, 1]
    xp, yp = im2_pts[:, 0], im2_pts[:, 1]

    A = np.zeros((2*N, 8), dtype=float)
    b = np.zeros((2*N,),   dtype=float)

    # x'-row
    A[0::2, 0] = x
    A[0::2, 1] = y
    A[0::2, 2] = 1
    A[0::2, 6] = -x * xp
    A[0::2, 7] = -y * xp
    b[0::2]    = xp

    # y'-row
    A[1::2, 3] = x
    A[1::2, 4] = y
    A[1::2, 5] = 1
    A[1::2, 6] = -x * yp
    A[1::2, 7] = -y * yp
    b[1::2]    = yp

    return A, b

def computeH(im1_pts: np.ndarray, im2_pts: np.ndarray) -> np.ndarray:
    """
    Solve for the 3x3 homography H such that p' ~ H p using the linear system Ah=b.
    H has 8 DoF with H[2,2]=1.

    Args:
        im1_pts: (N,2) source points (x,y)
        im2_pts: (N,2) destination points (x',y')
    Returns:
        H: (3,3) homography with H[2,2]=1
    """
    A, b = build_Ab(im1_pts, im2_pts)

    # Least squares (overdetermined when N>4)
    # We solve min ||A h - b||_2. This is stable and allowed by the spec.
    h, *_ = np.linalg.lstsq(A, b, rcond=None)  # h shape (8,)

    # Assemble H with bottom-right fixed to 1
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ], dtype=float)

    # Optional: normalize so H[2,2]=1 exactly (already is) and avoid scale drift.
    H = H / H[2, 2]

    return H

def reprojection_rmse(H: np.ndarray, im1_pts: np.ndarray, im2_pts: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute RMSE of projecting im1_pts with H and comparing to im2_pts.
    Returns (rmse, per_point_errors).
    """
    im1_pts = np.asarray(im1_pts, dtype=float)
    im2_pts = np.asarray(im2_pts, dtype=float)
    N = im1_pts.shape[0]

    P1 = np.c_[im1_pts, np.ones(N)]                 # (N,3)
    P2_hat = (H @ P1.T).T                           # (N,3)
    P2_hat = P2_hat[:, :2] / P2_hat[:, [2]]         # normalize
    err = np.sqrt(np.sum((P2_hat - im2_pts)**2, axis=1))
    rmse = float(np.sqrt(np.mean(err**2)))
    return rmse, err
