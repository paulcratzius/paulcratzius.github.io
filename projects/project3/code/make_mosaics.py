# projects/project3/code/make_mosaics.py
import json
import numpy as np
from pathlib import Path
from PIL import Image
from .warp import warpImageBilinear, _compute_canvas_bounds
from .compute_homography import computeH

def _read_H_txt(path):
    return np.loadtxt(path, dtype=float).reshape(3,3)

def _feather_alpha(h, w):
    # 1 in der Mitte, 0 am Rand (linear)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]   # (h,1)
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]   # (1,w)

    # broadcast auf (h,w)
    x_b = np.broadcast_to(x, (h, w))
    y_b = np.broadcast_to(y, (h, w))

    # Distanz zur nächsten Kante (pro Pixel)
    dist_edge = np.minimum(np.minimum(x_b, 1.0 - x_b),
                           np.minimum(y_b, 1.0 - y_b))

    # normalisieren auf [0,1]
    m = float(dist_edge.max()) if np.isfinite(dist_edge.max()) and dist_edge.max() > 0 else 1.0
    a = (dist_edge / m)[..., None]   # (h,w,1)

    return a.astype(np.float32)


def mosaic_three(center_path, left_path, right_path,
                 H_L_to_C, H_R_to_C, out_path, use_bilinear=True):
    C = Image.open(center_path).convert('RGB')
    L = Image.open(left_path).convert('RGB')
    R = Image.open(right_path).convert('RGB')

    # Bounds: projiziere Ecken von L und R in C-Koordinaten, plus C selbst
    xminL,yminL,xmaxL,ymaxL = _compute_canvas_bounds(np.array(L), H_L_to_C)
    xminR,yminR,xmaxR,ymaxR = _compute_canvas_bounds(np.array(R), H_R_to_C)
    xminC,yminC,xmaxC,ymaxC = 0,0,C.size[0]-1,C.size[1]-1

    xmin = min(xminL, xminR, xminC)
    ymin = min(yminL, yminR, yminC)
    xmax = max(xmaxL, xmaxR, xmaxC)
    ymax = max(ymaxL, ymaxR, ymaxC)

    # warp left/right
    warp = warpImageBilinear if use_bilinear else warpImageNearestNeighbor
    Lw, La = warp(L, H_L_to_C, (xmin, ymin, xmax, ymax))
    Rw, Ra = warp(R, H_R_to_C, (xmin, ymin, xmax, ymax))
    # place center on common canvas
    W = xmax - xmin + 1; H = ymax - ymin + 1
    Cw = np.zeros((H,W,3), dtype=np.float32); Ca = np.zeros((H,W,1), dtype=np.float32)
    x0 = -xmin; y0 = -ymin
    Cw[y0:y0+C.size[1], x0:x0+C.size[0]] = np.array(C, dtype=np.float32)  # careful H/W swap (PIL size = W,H)
    Ca[y0:y0+C.size[1], x0:x0+C.size[0], 0] = 1.0

    # Feather alphas
    for A in (La, Ra, Ca):
        h,w,_ = A.shape
        A[:] = np.minimum(A, _feather_alpha(h,w))  # ramp to edges

    # Weighted average
    num = Lw*La + Rw*Ra + Cw*Ca
    den = La + Ra + Ca + 1e-8
    out = num / den
    Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save(out_path)
    return out_path

def mosaic_from_json(scene_dir, out_dir, scene):
    scene_dir = Path(scene_dir)
    # Wir wollen <project3> als Root, nicht <project3>/outputs
    project_root = scene_dir.parents[1]            # .../project3
    img_dir = project_root / "images"

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    CL = img_dir / f"{scene}_center.jpg"
    LL = img_dir / f"{scene}_left.jpg"
    RR = img_dir / f"{scene}_right.jpg"

    # Debug: existieren die Bilder?
    for p in (CL, LL, RR):
        if not p.exists():
            raise FileNotFoundError(f"[mosaic] missing image: {p}")

    # Korrespondenzen lesen und H berechnen
    dL = json.load(open(scene_dir / f"{scene}_left_to_center.json"))
    dR = json.load(open(scene_dir / f"{scene}_right_to_center.json"))
    HL = computeH(np.array(dL["A"], float), np.array(dL["B"], float))
    HR = computeH(np.array(dR["A"], float), np.array(dR["B"], float))

    return mosaic_three(CL, LL, RR, HL, HR, out_dir / f"{scene}_mosaic.jpg")

