# projects/project3/code/main_mosaic_bwdist.py
# Build mosaics with distance-transform (bwdist) weighting.
# Uses: images in project3/images, homographies in project3/outputs/a2/H_*_to_C.txt

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter


# ----------------------------- Paths -----------------------------

# This file lives in .../projects/project3/code/
PROJECTS_DIR = Path(__file__).resolve().parents[2]          # .../projects
PROJ3        = PROJECTS_DIR / "project3"
IMAGES_DIR   = PROJ3 / "images"                              # <-- as requested
A2_DIR       = PROJ3 / "outputs" / "a2"
OUT_DIR      = PROJ3 / "outputs" / "a4_weighted" / "dist"

SCENES_DEFAULT = ["ihouse", "stadium_low", "stadium_high"]


# ------------------------------ IO --------------------------------

def load_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0

def save_rgb(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr*255, 0, 255).astype(np.uint8)).save(path)

def save_gray(arr01: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr01*255, 0, 255).astype(np.uint8), mode="L").save(path)

def load_H(scene: str, side: str) -> np.ndarray:
    """
    side in {'L','R'}; expects files:
      H_{scene}_L_to_C.txt
      H_{scene}_R_to_C.txt
    """
    p = A2_DIR / f"H_{scene}_{side}_to_C.txt"
    if not p.exists():
        raise FileNotFoundError(f"Homography not found: {p}")
    H = np.loadtxt(p).astype(np.float64)
    if H.shape != (3,3):
        raise ValueError(f"Invalid H shape in {p}: {H.shape}")
    return H


# --------------------------- Geometry -----------------------------

def corners_hw(h: int, w: int) -> np.ndarray:
    # 4 homogeneous corners (x,y,1)
    return np.array([[0,   0,   1],
                     [w-1, 0,   1],
                     [w-1, h-1, 1],
                     [0,   h-1, 1]], dtype=np.float64)

def apply_H(xy1: np.ndarray, H: np.ndarray) -> np.ndarray:
    uvw = xy1 @ H.T
    uv  = uvw[:, :2] / uvw[:, 2:3]
    return uv

def bilinear_sample(im: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    im: HxWxC (float32 0..1)
    Sample at (x,y) in im-coordinates. Returns (samples, good_mask).
    """
    H, W, C = im.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    good = (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)
    out  = np.zeros((x.size, C), dtype=np.float32)
    if not np.any(good):
        return out, good

    xg, yg = x[good], y[good]
    x0g, x1g = x0[good], x1[good]
    y0g, y1g = y0[good], y1[good]

    Ia = im[y0g, x0g, :]
    Ib = im[y0g, x1g, :]
    Ic = im[y1g, x0g, :]
    Id = im[y1g, x1g, :]

    wa = (x1g - xg) * (y1g - yg)
    wb = (xg - x0g) * (y1g - yg)
    wc = (x1g - xg) * (yg - y0g)
    wd = (xg - x0g) * (yg - y0g)

    out[good] = Ia*wa[:,None] + Ib*wb[:,None] + Ic*wc[:,None] + Id*wd[:,None]
    return out, good

def compute_canvas_and_transforms(C: np.ndarray, H_LC: np.ndarray, L: np.ndarray,
                                  H_RC: np.ndarray, R: np.ndarray):
    """
    Compute a canvas that contains: center (I), left warped by H_LC, right warped by H_RC.
    Returns: (outH,outW), T, Hc, Hl, Hr  where H* map src->canvas.
    """
    hC, wC = C.shape[:2]; hL, wL = L.shape[:2]; hR, wR = R.shape[:2]

    corC      = corners_hw(hC,wC)                # already in center coords
    corL_inC  = apply_H(corners_hw(hL,wL), H_LC) # left corners → center coords
    corR_inC  = apply_H(corners_hw(hR,wR), H_RC) # right corners → center coords

    all_pts = np.vstack([corC[:,:2], corL_inC, corR_inC])

    minx = np.floor(all_pts[:,0].min()).astype(int)
    miny = np.floor(all_pts[:,1].min()).astype(int)
    maxx = np.ceil( all_pts[:,0].max()).astype(int)
    maxy = np.ceil( all_pts[:,1].max()).astype(int)

    tx, ty = -minx, -miny
    T = np.array([[1,0,tx],
                  [0,1,ty],
                  [0,0, 1]], dtype=np.float64)

    outW = int(maxx - minx + 1)
    outH = int(maxy - miny + 1)

    Hc = T @ np.eye(3)
    Hl = T @ H_LC
    Hr = T @ H_RC
    return (outH, outW), T, Hc, Hl, Hr

def inverse_warp_to_canvas(src: np.ndarray, H_src2canvas: np.ndarray,
                           outH: int, outW: int):
    """
    Inverse warping mit bilinearer Interpolation.
    Gibt zurück: (warped RGB (H,W,3), gültige Maske (H,W))
    """
    Hinv = np.linalg.inv(H_src2canvas)
    ys, xs = np.meshgrid(np.arange(outH), np.arange(outW), indexing="ij")
    tgt = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3)

    src_xy = (tgt @ Hinv.T)
    src_xy = src_xy[:, :2] / src_xy[:, 2:3]
    x = src_xy[:, 0]
    y = src_xy[:, 1]

    # bilinear_sample liefert bereits ein (H*W,3)-Array und die bool-Maske (H*W,)
    samp, good = bilinear_sample(src, x, y)

    # Einfach reshapen – kein erneutes Zuweisen über good!
    out  = samp.reshape(outH, outW, 3)
    mask = good.reshape(outH, outW)

    return out, mask



# --------------------- Distance-based weighting ---------------------

def weight_from_distance(mask: np.ndarray, blur_sigma: float = 2.0, gamma: float = 1.0) -> np.ndarray:
    """
    mask: boolean or {0,1} array marking valid pixels of that warped image on the canvas.
    Returns a soft weight: distance to boundary, normalized to [0,1], optional Gaussian blur,
    and exponent gamma for shaping (gamma>1 center-peakier, gamma<1 flatter).
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    dist = distance_transform_edt(mask)
    if blur_sigma and blur_sigma > 0:
        dist = gaussian_filter(dist, blur_sigma, mode="nearest")
    if dist.max() > 0:
        dist = dist / dist.max()
    w = (dist ** gamma) * mask.astype(np.float32)
    return w

def normalize_weights(ws: list[np.ndarray]) -> list[np.ndarray]:
    Wsum = np.maximum(np.sum(ws, axis=0), 1e-8)
    return [w / Wsum for w in ws]


# ------------------------------ Pipeline -----------------------------

def run_scene(scene: str, sigma: float, gamma: float):
    print(f"[bwdist-mosaic] {scene}")

    # 1) Load inputs
    L = load_rgb(IMAGES_DIR / f"{scene}_left.jpg")
    C = load_rgb(IMAGES_DIR / f"{scene}_center.jpg")
    R = load_rgb(IMAGES_DIR / f"{scene}_right.jpg")

    # 2) Homographies: left->center, right->center
    H_LC = load_H(scene, "L")
    H_RC = load_H(scene, "R")

    # 3) Canvas & transforms (src -> canvas)
    (outH, outW), T, Hc, Hl, Hr = compute_canvas_and_transforms(C, H_LC, L, H_RC, R)

    # 4) Inverse warp to canvas
    Cc, Mc = inverse_warp_to_canvas(C, Hc, outH, outW)
    Lc, Ml = inverse_warp_to_canvas(L, Hl, outH, outW)
    Rc, Mr = inverse_warp_to_canvas(R, Hr, outH, outW)

    # Save warped + masks
    wdir = OUT_DIR / "warped" / scene
    save_rgb(Cc, wdir / "center_warped.jpg")
    save_rgb(Lc, wdir / "left_warped.jpg")
    save_rgb(Rc, wdir / "right_warped.jpg")
    save_gray(Mc.astype(np.float32), wdir / "center_mask.png")
    save_gray(Ml.astype(np.float32), wdir / "left_mask.png")
    save_gray(Mr.astype(np.float32), wdir / "right_mask.png")

    # 5) Distance-transform weights (like MATLAB bwdist)
    w1 = weight_from_distance(Ml, blur_sigma=sigma, gamma=gamma)  # left
    w2 = weight_from_distance(Mc, blur_sigma=sigma, gamma=gamma)  # center
    w3 = weight_from_distance(Mr, blur_sigma=sigma, gamma=gamma)  # right
    w1, w2, w3 = normalize_weights([w1, w2, w3])

    # Save weights
    wv = OUT_DIR / "weights" / scene
    save_gray(w1, wv / "weight_w1_left.png")
    save_gray(w2, wv / "weight_w2_center.png")
    save_gray(w3, wv / "weight_w3_right.png")

    # 6) Blend
    out = Lc * w1[...,None] + Cc * w2[...,None] + Rc * w3[...,None]
    save_rgb(out, OUT_DIR / f"{scene}_mosaic_dist.jpg")
    print(f"  -> saved: {OUT_DIR / f'{scene}_mosaic_dist.jpg'}")


# -------------------------------- Main --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Distance-transform (bwdist) weighted mosaics.")
    ap.add_argument("--scenes", type=str, default=",".join(SCENES_DEFAULT),
                    help="comma-separated list (default: ihouse,stadium_low,stadium_high)")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="Gaussian blur on distance map (px). 0 disables.")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Exponent shaping (0.7 flatter, 1 neutral, >1 center-peakier)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    for sc in scenes:
        run_scene(sc, sigma=args.sigma, gamma=args.gamma)

    print("\n[done] distance-weighted mosaics written to:", OUT_DIR)


if __name__ == "__main__":
    main()
