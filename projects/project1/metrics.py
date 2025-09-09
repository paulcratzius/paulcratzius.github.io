# metrics.py
import numpy as np

def crop_inner(img: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """
    Schneidet insgesamt frac (z.B. 0.10 = 10%) ab: je Seite frac/2.
    Funktioniert für 2D (H,W) und 3D (H,W,C).
    """
    H, W = img.shape[:2]
    dh = int((frac/2) * H)
    dw = int((frac/2) * W)
    if img.ndim == 2:
        return img[dh:H-dh, dw:W-dw]
    else:
        return img[dh:H-dh, dw:W-dw, :]

def overlap_views(ref: np.ndarray, tgt: np.ndarray, dy: int, dx: int):
    """
    Liefert Slices der überlappenden Bereiche, ohne Wraparound.
    """
    H, W = ref.shape[:2]
    y0r = max(0, dy);      y1r = min(H, H+dy)
    x0r = max(0, dx);      x1r = min(W, W+dx)

    y0t = max(0, -dy);     y1t = min(H, H-dy)
    x0t = max(0, -dx);     x1t = min(W, W-dx)

    if y1r - y0r <= 0 or x1r - x0r <= 0:
        return None, None
    return (slice(y0r, y1r), slice(x0r, x1r)), (slice(y0t, y1t), slice(x0t, x1t))

def ssd(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sum(d*d))

def ncc(a: np.ndarray, b: np.ndarray) -> float:
    A = a - a.mean()
    B = b - b.mean()
    denom = (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12)
    return float((A * B).sum() / denom)

def phase_correlation(im1: np.ndarray, im2: np.ndarray):
    """
    Low-level Phase Correlation (nur numpy.fft).
    Gibt Integer-(dy, dx) zurück.
    """
    F1 = np.fft.fft2(im1)
    F2 = np.fft.fft2(im2)
    R = F1 * np.conj(F2)
    R /= (np.abs(R) + 1e-8)
    r = np.fft.ifft2(R)
    r = np.abs(r)
    y, x = np.unravel_index(np.argmax(r), r.shape)

    H, W = im1.shape[:2]
    if y > H // 2: y -= H
    if x > W // 2: x -= W
    return int(y), int(x)
