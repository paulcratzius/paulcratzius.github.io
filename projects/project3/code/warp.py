# projects/project3/code/warp.py
import numpy as np
from PIL import Image

def _to_numpy_rgb(im):
    if isinstance(im, Image.Image):
        im = np.array(im)
    im = im.astype(np.float32)
    if im.ndim == 2:
        im = im[..., None]
    return im  # HxWxC float32

def _corners_hw(H, W):
    return np.array([[0,0,1],[W-1,0,1],[0,H-1,1],[W-1,H-1,1]], dtype=np.float32).T  # 3x4

def _project_points(H, pts3xN):
    X = H @ pts3xN
    X /= X[2:3]
    return X[:2]  # 2xN

def _compute_canvas_bounds(img, H):
    Hsrc, Wsrc = img.shape[0], img.shape[1]
    crn = _corners_hw(Hsrc, Wsrc)          # 3x4 corners in src
    dst = _project_points(H, crn)          # 2x4 in dst
    minx = np.floor(dst[0].min()).astype(int)
    maxx = np.ceil (dst[0].max()).astype(int)
    miny = np.floor(dst[1].min()).astype(int)
    maxy = np.ceil (dst[1].max()).astype(int)
    return minx, miny, maxx, maxy

def _alpha_channel(h, w):
    # simple 1 at valid pixels, 0 elsewhere (will be refined for blending later)
    return np.ones((h, w, 1), dtype=np.float32)

def warpImageNearestNeighbor(im, H, out_bounds=None, return_alpha=True):
    """
    Inverse warping: for every output pixel x', backproject with H^{-1} to source.
    NN interpolation (round).
    Args:
      im: PIL Image or np.array (H,W,[C])
      H: 3x3 homography mapping src->dst (we create dst canvas automatically)
      out_bounds: optional (xmin, ymin, xmax, ymax) in dst coords
    Returns: warped image (float32 in [0,255]) and optional alpha
    """
    src = _to_numpy_rgb(im)
    Hinv = np.linalg.inv(H).astype(np.float32)

    if out_bounds is None:
        xmin, ymin, xmax, ymax = _compute_canvas_bounds(src, H)
    else:
        xmin, ymin, xmax, ymax = map(int, out_bounds)

    Wout = xmax - xmin + 1
    Hout = max(0, ymax - ymin + 1)
    out = np.zeros((Hout, Wout, src.shape[2]), dtype=np.float32)
    alpha = np.zeros((Hout, Wout, 1), dtype=np.float32)

    # grid of output coords (x',y')
    xs = np.arange(xmin, xmax+1, dtype=np.float32)
    ys = np.arange(ymin, ymax+1, dtype=np.float32)
    Xp, Yp = np.meshgrid(xs, ys)                    # Hout x Wout
    ones = np.ones_like(Xp)
    dst_h = np.stack([Xp, Yp, ones], axis=0).reshape(3,-1)  # 3 x (Hout*Wout)

    # backproject to source
    src_h = Hinv @ dst_h
    src_h /= src_h[2:3]
    X = src_h[0].reshape(Hout, Wout)
    Y = src_h[1].reshape(Hout, Wout)

    # NN indices
    Xi = np.rint(X).astype(int)
    Yi = np.rint(Y).astype(int)

    # valid mask
    mask = (Xi>=0)&(Yi>=0)&(Xi<src.shape[1])&(Yi<src.shape[0])

    for c in range(src.shape[2]):
        out[...,c][mask] = src[Yi[mask], Xi[mask], c]
    alpha[mask, 0] = 1.0

    if src.shape[2]==1:
        out = out[...,0]

    return (out, alpha) if return_alpha else out

def warpImageBilinear(im, H, out_bounds=None, return_alpha=True):
    """
    Inverse warping with bilinear interpolation.
    """
    src = _to_numpy_rgb(im)
    Hinv = np.linalg.inv(H).astype(np.float32)

    if out_bounds is None:
        xmin, ymin, xmax, ymax = _compute_canvas_bounds(src, H)
    else:
        xmin, ymin, xmax, ymax = map(int, out_bounds)

    Wout = xmax - xmin + 1
    Hout = max(0, ymax - ymin + 1)
    out = np.zeros((Hout, Wout, src.shape[2]), dtype=np.float32)
    alpha = np.zeros((Hout, Wout, 1), dtype=np.float32)

    xs = np.arange(xmin, xmax+1, dtype=np.float32)
    ys = np.arange(ymin, ymax+1, dtype=np.float32)
    Xp, Yp = np.meshgrid(xs, ys)
    ones = np.ones_like(Xp)
    dst_h = np.stack([Xp, Yp, ones], axis=0).reshape(3,-1)

    src_h = Hinv @ dst_h
    src_h /= src_h[2:3]
    X = src_h[0].reshape(Hout, Wout)
    Y = src_h[1].reshape(Hout, Wout)

    x0 = np.floor(X).astype(int); x1 = x0 + 1
    y0 = np.floor(Y).astype(int); y1 = y0 + 1

    wx = X - x0; wy = Y - y0
    w00 = (1-wx)*(1-wy); w10 = wx*(1-wy); w01 = (1-wx)*wy; w11 = wx*wy

    valid = (x0>=0)&(y0>=0)&(x1<src.shape[1])&(y1<src.shape[0])

    for c in range(src.shape[2]):
        I00 = np.zeros_like(X); I10 = np.zeros_like(X)
        I01 = np.zeros_like(X); I11 = np.zeros_like(X)
        I00[valid] = src[y0[valid], x0[valid], c]
        I10[valid] = src[y0[valid], x1[valid], c]
        I01[valid] = src[y1[valid], x0[valid], c]
        I11[valid] = src[y1[valid], x1[valid], c]
        out[...,c] = I00*w00 + I10*w10 + I01*w01 + I11*w11

    alpha[valid, 0] = 1.0
    if src.shape[2]==1:
        out = out[...,0]
    return (out, alpha) if return_alpha else out
