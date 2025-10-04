#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from PIL import Image

from .warp import warpImageBilinear, _compute_canvas_bounds
from .compute_homography import computeH


# --- NEW: bwdist-style weights (Distance Transform + blur + gamma) ---
def _weights_bwdist(La, Ca, Ra, center_boost=1.10, sigma=25, gamma=1.0):
    """
    La, Ca, Ra: (H,W,1) Binär-Alpha (0/1) auf gemeinsamer Canvas.
    center_boost: Center leicht bevorzugen (z.B. 1.10).
    sigma: Gauß-Blur auf der Distanzkarte (Pixel). 20–40 wirkt gut.
    gamma: Nichtlinearität (<=1 weicher, >1 steiler). 0.8–1.2 passt gut.
    """
    import numpy as np
    try:
        from scipy.ndimage import distance_transform_edt as edt, gaussian_filter
    except Exception as e:
        raise RuntimeError(
            "Für bwdist-Gewichte brauchst du SciPy. Installiere im viz-Env: "
            "`pip install scipy`"
        ) from e

    def dt_weight(A):
        # 1) Euklidische Distanz *innerhalb* der gültigen Region
        M = (A[..., 0] > 0.5)
        d = edt(M).astype(np.float32)
        # 2) Großflächig weich zeichnen wie im Video
        if sigma and sigma > 0:
            d = gaussian_filter(d, sigma=float(sigma))
        # 3) Auf [0,1] normieren (nur dort, wo Maske 1 ist)
        if np.any(M):
            d = d / (d[M].max() + 1e-8)
        # 4) Sanfte Nichtlinearität (gamma)
        if gamma and gamma != 1.0:
            d = np.power(d, float(gamma))
        return d[..., None].astype(np.float32)

    wL = dt_weight(La)
    wC = dt_weight(Ca) * float(center_boost)
    wR = dt_weight(Ra)

    den = wL + wC + wR + 1e-8
    return wL/den, wC/den, wR/den



# ---------- Gewichte: Hann/Cosine ----------
def _hann2d(h, w):
    y = np.hanning(max(h, 3)).astype(np.float32)[:, None]
    x = np.hanning(max(w, 3)).astype(np.float32)[None, :]
    return (y / y.max()) * (x / x.max())  # (h,w)

def _weights_hann(La, Ca, Ra, center_boost=1.10):
    h, w = La.shape[:2]
    base = _hann2d(h, w)[..., None]  # (h,w,1)
    wL = base * La
    wC = base * Ca * center_boost
    wR = base * Ra
    den = wL + wC + wR + 1e-8
    return wL/den, wC/den, wR/den

# ---------- Gewichte: Distance Transform (wie bwdist) ----------
def _edt01(mask01):
    """Distance zur nächstgelegenen Kante *innerhalb* der Maske, normalisiert nach [0,1]."""
    M = (mask01[..., 0] > 0.5)
    try:
        from scipy.ndimage import distance_transform_edt as edt
        d = edt(M)
    except Exception:
        # Fallback: Hann, falls SciPy nicht installiert ist
        h, w = M.shape
        d = _hann2d(h, w) * M
    if np.any(M):
        d = d / (d[M].max() + 1e-8)
    return d[..., None].astype(np.float32)

def _weights_dist(La, Ca, Ra, center_boost=1.10):
    dL = _edt01(La)
    dC = _edt01(Ca) * center_boost
    dR = _edt01(Ra)
    den = dL + dC + dR + 1e-8
    return dL/den, dC/den, dR/den

# ---------- Visualisierung der drei Gewichte ----------
def save_gray01(arr01, path: Path):
    arr = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr.squeeze(-1)).save(path)

def visualize_three_weights(out_dir: Path, wL, wC, wR, prefix=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_gray01(wL, out_dir / f"{prefix}weight_w1_left.png")
    save_gray01(wC, out_dir / f"{prefix}weight_w2_center.png")
    save_gray01(wR, out_dir / f"{prefix}weight_w3_right.png")

# ---------- Mosaic-Builder mit frei wählbaren Gewichten ----------
def mosaic_three_weighted(center_path, left_path, right_path,
                          H_L_to_C, H_R_to_C,
                          out_img_path: Path,
                          weight_mode: str = "hann",
                          center_boost: float = 1.10,
                          return_weights_dir: Path | None = None):
    """
    weight_mode: 'hann' (raised-cosine) oder 'dist' (distance transform)
    """
    C = Image.open(center_path).convert('RGB')
    L = Image.open(left_path).convert('RGB')
    R = Image.open(right_path).convert('RGB')

    # gemeinsame Canvas-Grenzen in Center-Koordinaten
    xminL,yminL,xmaxL,ymaxL = _compute_canvas_bounds(np.array(L), H_L_to_C)
    xminR,yminR,xmaxR,ymaxR = _compute_canvas_bounds(np.array(R), H_R_to_C)
    xminC,yminC,xmaxC,ymaxC = 0,0,C.size[0]-1,C.size[1]-1

    xmin = min(xminL, xminR, xminC)
    ymin = min(yminL, yminR, yminC)
    xmax = max(xmaxL, xmaxR, xmaxC)
    ymax = max(ymaxL, ymaxR, ymaxC)

    # warp left/right auf die gemeinsame Leinwand
    Lw, La = warpImageBilinear(L, H_L_to_C, (xmin, ymin, xmax, ymax))
    Rw, Ra = warpImageBilinear(R, H_R_to_C, (xmin, ymin, xmax, ymax))

    # Center einlegen
    W = xmax - xmin + 1; Hh = ymax - ymin + 1
    Cw = np.zeros((Hh, W, 3), dtype=np.float32)
    Ca = np.zeros((Hh, W, 1), dtype=np.float32)
    Wc, Hc = C.size
    x0 = -xmin; y0 = -ymin
    Cw[y0:y0+Hc, x0:x0+Wc, :] = np.array(C, dtype=np.float32)
    Ca[y0:y0+Hc, x0:x0+Wc, 0] = 1.0

    # Gewichte wählen
    if weight_mode == "hann":
        wL, wC, wR = _weights_hann(La, Ca, Ra, center_boost=center_boost)
    elif weight_mode == "dist":
        wL, wC, wR = _weights_dist(La, Ca, Ra, center_boost=center_boost)
    elif weight_mode == "bwdist":  # NEW
        wL, wC, wR = _weights_bwdist(La, Ca, Ra,
                                    center_boost=center_boost,
                                    sigma=25,      # probiere 20–40
                                    gamma=0.9)     # 0.8–1.0 ergibt sehr weiche Übergänge
    else:
        raise ValueError("weight_mode must be 'hann', 'dist', or 'bwdist'")


    den = wL + wC + wR + 1e-8
    out = (Lw * wL + Cw * wC + Rw * wR) / den

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(out_img_path)
    return out_img_path
