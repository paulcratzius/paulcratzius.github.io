# projects/project3/code/main_mosaic_bwdist_boundary.py
# Distance-to-boundary (explicit boundary) weighting for mosaics.
# Bilder:   projects/project3/images/{scene}_{left|center|right}.jpg
# Homogs:   projects/project3/outputs/a2/H_{scene}_{L|R}_to_C.txt
# Outputs:  projects/project3/outputs/a4_weighted/boundary_dist/...

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_erosion

# ----------------------------- Pfade -----------------------------
PROJ_ROOT   = Path(__file__).resolve().parents[2]      # .../projects
PROJ3       = PROJ_ROOT / "project3"
IMAGES_DIR  = PROJ3 / "images"
A2_DIR      = PROJ3 / "outputs" / "a2"
OUT_DIR     = PROJ3 / "outputs" / "a4_weighted" / "boundary_dist"

SCENES_DEFAULT = ["ihouse", "stadium_low", "stadium_high"]

# ------------------------------ IO --------------------------------
def load_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0

def save_rgb(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr*255, 0, 255).astype(np.uint8)).save(path)

def save_gray(arr01: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr01*255, 0, 255).astype(np.uint8), mode="L").save(path)

def load_H(scene: str, side: str) -> np.ndarray:
    p = A2_DIR / f"H_{scene}_{side}_to_C.txt"
    H = np.loadtxt(p).astype(np.float64)
    if H.shape != (3,3):
        raise ValueError(f"Invalid homography shape in {p}: {H.shape}")
    return H

# --------------------------- Geometrie ----------------------------
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

    Ia = im[y0g, x0g]; Ib = im[y0g, x1g]
    Ic = im[y1g, x0g]; Id = im[y1g, x1g]

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

    outW = maxx-minx+1; outH = maxy-miny+1
    Hc = T @ np.eye(3); Hl = T @ H_LC; Hr = T @ H_RC
    return (outH, outW), T, Hc, Hl, Hr

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

# --------------- EXPLIZITE Boundary-Weight-Funktion ----------------
def boundary_from_mask(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """
    Boundary = äußerste weiße Pixel: mask & ~erode(mask)
    """
    m = mask.astype(bool)
    er = binary_erosion(m, iterations=iters, border_value=0)
    boundary = m & (~er)
    return boundary

def weight_from_distance_to_boundary(mask: np.ndarray,
                                     blur_sigma: float = 0.0,
                                     gamma: float = 1.0) -> np.ndarray:
    """
    Gewichte innerhalb der weißen Pixel proportional zur Distanz zur (expliziten) Boundary.
    - Boundary wird als mask & ~erode(mask) definiert (äußerste weiße Pixel).
    - Wir bauen ein Hilfsbild 'invB' mit 0 auf der Boundary, 1 sonst; EDT misst Distanz zu 0 → zur Boundary.
    - Danach beschränken wir auf das Innere (mask), normalisieren auf [0,1], optional weichzeichnen & Gamma.
    """
    m = mask.astype(bool)
    if not np.any(m):  # leer
        return np.zeros_like(mask, dtype=np.float32)

    B = boundary_from_mask(m, iters=1)      # True genau auf Boundary (neue Kante)
    invB = ~B                               # 0 an Boundary, 1 sonst
    d = distance_transform_edt(invB)        # Distanz zur nächstgelegenen Boundary
    d = d * m                               # nur innerhalb der weißen Menge
    if blur_sigma and blur_sigma > 0:
        d = gaussian_filter(d.astype(np.float32), blur_sigma, mode="nearest")

    dmax = d[m].max() if np.any(m) else 0.0
    if dmax > 0:
        d = d / dmax
    w = (d ** gamma) * m
    return w.astype(np.float32)

def normalize_weights(ws: list[np.ndarray]) -> list[np.ndarray]:
    Wsum = np.maximum(np.sum(ws, axis=0), 1e-8)
    return [w / Wsum for w in ws]

# ------------------------------ Pipeline ----------------------------
def run_scene(scene: str, sigma: float, gamma: float):
    print(f"[boundary-dist mosaic] {scene}")

    # 1) Laden
    L = load_rgb(IMAGES_DIR / f"{scene}_left.jpg")
    C = load_rgb(IMAGES_DIR / f"{scene}_center.jpg")
    R = load_rgb(IMAGES_DIR / f"{scene}_right.jpg")

    # 2) Homographien (left/right → center)
    H_LC = load_H(scene, "L")
    H_RC = load_H(scene, "R")

    # 3) Canvas & Warps
    (outH, outW), T, Hc, Hl, Hr = compute_canvas_and_transforms(C, H_LC, L, H_RC, R)
    Cc, Mc = inverse_warp_to_canvas(C, Hc, outH, outW)
    Lc, Ml = inverse_warp_to_canvas(L, Hl, outH, outW)
    Rc, Mr = inverse_warp_to_canvas(R, Hr, outH, outW)

    # 4) Debug speichern: Warps & Masken
    wdir = OUT_DIR / "warped" / scene
    save_rgb(Cc, wdir / "center_warped.jpg")
    save_rgb(Lc, wdir / "left_warped.jpg")
    save_rgb(Rc, wdir / "right_warped.jpg")
    save_gray(Mc.astype(np.float32), wdir / "center_mask.png")
    save_gray(Ml.astype(np.float32), wdir / "left_mask.png")
    save_gray(Mr.astype(np.float32), wdir / "right_mask.png")

    # 5) Boundary-Maps (Visualisierung)
    bdir = OUT_DIR / "boundary" / scene
    save_gray(boundary_from_mask(Ml).astype(np.float32), bdir / "left_boundary.png")
    save_gray(boundary_from_mask(Mc).astype(np.float32), bdir / "center_boundary.png")
    save_gray(boundary_from_mask(Mr).astype(np.float32), bdir / "right_boundary.png")

    # 6) Distanz-zu-Boundary Gewichte
    wL = weight_from_distance_to_boundary(Ml, blur_sigma=sigma, gamma=gamma)
    wC = weight_from_distance_to_boundary(Mc, blur_sigma=sigma, gamma=gamma)
    wR = weight_from_distance_to_boundary(Mr, blur_sigma=sigma, gamma=gamma)

    # 7) Per-Pixel Normalisierung (nur wo mind. ein Bild beiträgt)
    wL, wC, wR = normalize_weights([wL, wC, wR])

    # 8) Gewichte speichern (Graustufen)
    wv = OUT_DIR / "weights" / scene
    save_gray(wL, wv / "weight_left.png")
    save_gray(wC, wv / "weight_center.png")
    save_gray(wR, wv / "weight_right.png")

    # 9) Blend
    out = Lc * wL[...,None] + Cc * wC[...,None] + Rc * wR[...,None]
    save_rgb(out, OUT_DIR / f"{scene}_mosaic_boundary_dist.jpg")
    print(f"  -> saved: {OUT_DIR / f'{scene}_mosaic_boundary_dist.jpg'}")

# -------------------------------- Main --------------------------------
def main():
    ap = argparse.ArgumentParser(description="Mosaics with explicit boundary distance weights.")
    ap.add_argument("--scenes", type=str, default=",".join(SCENES_DEFAULT),
                    help="comma-separated scenes (default: ihouse,stadium_low,stadium_high)")
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="Gaussian blur on the distance map (px). Use 1–3 to soften sharp corners.")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Exponent shaping (<1 flatter, >1 center-dominant).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    for sc in scenes:
        run_scene(sc, sigma=args.sigma, gamma=args.gamma)

    print("\n[done] boundary-distance mosaics written to:", OUT_DIR)

if __name__ == "__main__":
    main()
