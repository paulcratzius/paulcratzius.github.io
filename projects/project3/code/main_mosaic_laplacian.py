# projects/project3/code/main_mosaic_laplacian.py
# Build mosaics using a Laplacian-pyramid blend (stack-based, no downsampling).
# Inputs:
#   images: projects/project3/images/{scene}_{left|center|right}.jpg
#   homogs: projects/project3/outputs/a2/H_{scene}_{L|R}_to_C.txt
# Outputs:
#   projects/project3/outputs/a4_weighted/laplacian/<scene>_*.jpg

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt,
    binary_erosion,
)

# ------------------------- Paths & defaults -------------------------

PROJ_ROOT   = Path(__file__).resolve().parents[2]  # .../projects
PROJ3       = PROJ_ROOT / "project3"
IMAGES_DIR  = PROJ3 / "images"
A2_DIR      = PROJ3 / "outputs" / "a2"
OUT_DIR     = PROJ3 / "outputs" / "a4_weighted" / "laplacian"

SCENES_DEFAULT = ["ihouse", "stadium_low", "stadium_high"]


# ------------------------------ IO ---------------------------------

def load_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0

def save_rgb(arr01: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def save_gray(arr01: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

def load_H(scene: str, side: str) -> np.ndarray:
    p = A2_DIR / f"H_{scene}_{side}_to_C.txt"
    H = np.loadtxt(p).astype(np.float64)
    if H.shape != (3,3):
        raise ValueError(f"Invalid homography shape in {p}: {H.shape}")
    return H


# --------------------------- Geometry/warp --------------------------

def corners_hw(h: int, w: int) -> np.ndarray:
    return np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]], dtype=np.float64)

def apply_H(xy1: np.ndarray, H: np.ndarray) -> np.ndarray:
    uvw = xy1 @ H.T
    return uvw[:, :2] / uvw[:, 2:3]

def bilinear_sample(im: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W, C = im.shape
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1;                         y1 = y0 + 1
    good = (x0>=0)&(y0>=0)&(x1<W)&(y1<H)
    out = np.zeros((x.size, C), dtype=np.float32)
    if not np.any(good):
        return out, good
    xg, yg = x[good], y[good]
    x0g, x1g, y0g, y1g = x0[good], x1[good], y0[good], y1[good]
    Ia = im[y0g, x0g]; Ib = im[y0g, x1g]; Ic = im[y1g, x0g]; Id = im[y1g, x1g]
    wa = (x1g - xg) * (y1g - yg)
    wb = (xg - x0g) * (y1g - yg)
    wc = (x1g - xg) * (yg - y0g)
    wd = (xg - x0g) * (yg - y0g)
    out[good] = Ia*wa[:,None] + Ib*wb[:,None] + Ic*wc[:,None] + Id*wd[:,None]
    return out, good

def compute_canvas_and_transforms(C: np.ndarray, H_LC: np.ndarray, L: np.ndarray,
                                  H_RC: np.ndarray, R: np.ndarray):
    hC,wC = C.shape[:2]; hL,wL = L.shape[:2]; hR,wR = R.shape[:2]
    corC     = corners_hw(hC,wC)
    corL_inC = apply_H(corners_hw(hL,wL), H_LC)
    corR_inC = apply_H(corners_hw(hR,wR), H_RC)
    all_pts = np.vstack([corC[:,:2], corL_inC, corR_inC])

    minx = int(np.floor(all_pts[:,0].min())); maxx = int(np.ceil(all_pts[:,0].max()))
    miny = int(np.floor(all_pts[:,1].min())); maxy = int(np.ceil(all_pts[:,1].max()))
    tx, ty = -minx, -miny
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)

    outW = maxx - minx + 1; outH = maxy - miny + 1
    Hc = T @ np.eye(3); Hl = T @ H_LC; Hr = T @ H_RC
    return (outH, outW), Hc, Hl, Hr

def inverse_warp_to_canvas(src: np.ndarray, H_src2canvas: np.ndarray, outH: int, outW: int):
    Hinv = np.linalg.inv(H_src2canvas)
    ys, xs = np.meshgrid(np.arange(outH), np.arange(outW), indexing="ij")
    tgt = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1,3)
    src_xy = tgt @ Hinv.T
    src_xy = src_xy[:, :2] / src_xy[:, 2:3]
    x, y = src_xy[:,0], src_xy[:,1]
    samp, good = bilinear_sample(src, x, y)
    out  = samp.reshape(outH, outW, 3)
    mask = good.reshape(outH, outW)
    return out, mask


# ------------------- Distance weights & boundaries ------------------

def boundary_from_mask(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = mask.astype(bool)
    er = binary_erosion(m, iterations=iters, border_value=0)
    return m & (~er)

def weight_from_distance_to_boundary(mask: np.ndarray,
                                     blur_sigma: float = 0.0,
                                     gamma: float = 1.0) -> np.ndarray:
    m = mask.astype(bool)
    if not np.any(m):
        return np.zeros_like(mask, dtype=np.float32)
    B = boundary_from_mask(m, iters=1)  # boundary = outermost white pixels
    invB = ~B                           # zeros at boundary
    d = distance_transform_edt(invB)    # distance to boundary
    d = d * m
    if blur_sigma and blur_sigma > 0:
        d = gaussian_filter(d.astype(np.float32), blur_sigma, mode="nearest")
    dmax = d[m].max()
    if dmax > 0:
        d = d / dmax
    w = (d ** gamma) * m
    return w.astype(np.float32)

def normalize_weights(ws: list[np.ndarray]) -> list[np.ndarray]:
    Wsum = np.maximum(np.sum(ws, axis=0), 1e-8)
    return [w / Wsum for w in ws]


# -------------------- Laplacian stacks & blending -------------------

def gaussian_stack(img: np.ndarray, levels: int, sigma: float) -> list[np.ndarray]:
    """No downsampling: repeated Gaussian blur."""
    gs = [img.astype(np.float32)]
    for _ in range(1, levels):
        gs.append(gaussian_filter(gs[-1], sigma=(sigma, sigma, 0) if img.ndim==3 else sigma))
    return gs

def laplacian_stack(img: np.ndarray, levels: int, sigma: float) -> list[np.ndarray]:
    """Laplacian stack (band-pass) + final low-pass; no downsampling."""
    G = gaussian_stack(img, levels, sigma)
    L = []
    for k in range(levels-1):
        L.append(G[k] - G[k+1])
    L.append(G[-1])  # base low-pass
    return L

def laplacian_blend(A: np.ndarray, B: np.ndarray, M: np.ndarray,
                    levels: int = 4, sigma: float = 2.0) -> np.ndarray:
    """
    Stack-based Laplacian blend:
      - A,B: RGB in [0,1]
      - M:  mask in [0,1] (white→take A, black→take B)
    """
    A = np.clip(A,0,1).astype(np.float32)
    B = np.clip(B,0,1).astype(np.float32)
    M = np.clip(M,0,1).astype(np.float32)

    LA = laplacian_stack(A, levels, sigma)
    LB = laplacian_stack(B, levels, sigma)

    # Gaussian mask stack (scalar per level)
    GM = gaussian_stack(M, levels, sigma)

    out = np.zeros_like(A, dtype=np.float32)
    # sum of blended bands
    for k in range(levels-1):
        Mk = GM[k][..., None]
        out += Mk * LA[k] + (1.0 - Mk) * LB[k]
    # add final low-pass
    Mk = GM[-1][..., None]
    out += Mk * LA[-1] + (1.0 - Mk) * LB[-1]
    return np.clip(out, 0, 1)


# ------------------------------- Pipeline --------------------------

def run_scene(scene: str, levels: int, sigma: float, dist_sigma: float, gamma: float):
    print(f"[laplacian-mosaic] {scene}")

    # 1) Load images
    L = load_rgb(IMAGES_DIR / f"{scene}_left.jpg")
    C = load_rgb(IMAGES_DIR / f"{scene}_center.jpg")
    R = load_rgb(IMAGES_DIR / f"{scene}_right.jpg")

    # 2) Load homographies (left/right -> center)
    H_LC = load_H(scene, "L")
    H_RC = load_H(scene, "R")

    # 3) Canvas & warp all three into a shared canvas (inverse warping, bilinear)
    (outH, outW), Hc, Hl, Hr = compute_canvas_and_transforms(C, H_LC, L, H_RC, R)
    Cc, Mc = inverse_warp_to_canvas(C, Hc, outH, outW)
    Lc, Ml = inverse_warp_to_canvas(L, Hl, outH, outW)
    Rc, Mr = inverse_warp_to_canvas(R, Hr, outH, outW)

    # Save warps/masks for reference
    wdir = OUT_DIR / "warped" / scene
    save_rgb(Cc, wdir / "center_warped.jpg")
    save_rgb(Lc, wdir / "left_warped.jpg")
    save_rgb(Rc, wdir / "right_warped.jpg")
    save_gray(Mc.astype(np.float32), wdir / "center_mask.png")
    save_gray(Ml.astype(np.float32), wdir / "left_mask.png")
    save_gray(Mr.astype(np.float32), wdir / "right_mask.png")

    # 4) Distance-to-boundary weights per warped image
    wL = weight_from_distance_to_boundary(Ml, blur_sigma=dist_sigma, gamma=gamma)
    wC = weight_from_distance_to_boundary(Mc, blur_sigma=dist_sigma, gamma=gamma)
    wR = weight_from_distance_to_boundary(Mr, blur_sigma=dist_sigma, gamma=gamma)

    # 5) Combine Left & Right first (normalized within L/R support)
    LR_denom = np.maximum(wL + wR, 1e-8)
    wL_norm = wL / LR_denom
    wR_norm = wR / LR_denom
    LR = Lc * wL_norm[...,None] + Rc * wR_norm[...,None]

    # Optional: save the intermediate L⊕R and weights
    idir = OUT_DIR / "intermediate" / scene
    save_rgb(LR, idir / "LR_blend_feather.jpg")
    save_gray(wL, OUT_DIR / "weights" / scene / "left_weight.png")
    save_gray(wC, OUT_DIR / "weights" / scene / "center_weight.png")
    save_gray(wR, OUT_DIR / "weights" / scene / "right_weight.png")

    # 6) Main Laplacian blend: A = Center, B = (L⊕R), mask = normalized center weight
    denom_AB = np.maximum(wC + (1.0 - wC), 1e-8)  # = 1, nur der Form halber
    M = np.clip(wC / denom_AB, 0, 1)              # in [0,1]

    out_lap = laplacian_blend(A=Cc, B=LR, M=M, levels=levels, sigma=sigma)
    save_rgb(out_lap, OUT_DIR / f"{scene}_mosaic_lap{levels}.jpg")

    # 7) Feather baseline (for comparison, same weights normalized global)
    wL_, wC_, wR_ = normalize_weights([wL, wC, wR])
    out_feather = Lc * wL_[...,None] + Cc * wC_[...,None] + Rc * wR_[...,None]
    save_rgb(out_feather, OUT_DIR / f"{scene}_mosaic_feather.jpg")

    print(f"  -> saved: {OUT_DIR / f'{scene}_mosaic_lap{levels}.jpg'}")

# --------------------------------- Main ---------------------------------

def main():
    ap = argparse.ArgumentParser(description="Laplacian (stack) blending for mosaics.")
    ap.add_argument("--scenes", type=str, default=",".join(SCENES_DEFAULT),
                    help="comma-separated scene list")
    ap.add_argument("--levels", type=int, default=4,
                    help="number of Laplacian levels (>=2)")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="Gaussian sigma for stacks (px)")
    ap.add_argument("--dist_sigma", type=float, default=2.0,
                    help="Gaussian sigma for distance-to-boundary weights")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="gamma shaping for distance weights")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    for sc in scenes:
        run_scene(sc,
                  levels=max(2, args.levels),
                  sigma=args.sigma,
                  dist_sigma=args.dist_sigma,
                  gamma=args.gamma)

    print("\n[done] Laplacian mosaics written to:", OUT_DIR)

if __name__ == "__main__":
    main()
