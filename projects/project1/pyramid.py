# pyramid.py
import numpy as np
from metrics import crop_inner, overlap_views, ncc, ssd, phase_correlation

def crop_borders(img: np.ndarray, frac: float = 0.10) -> np.ndarray:
    H, W = img.shape[:2]
    dh = int((frac/2) * H)
    dw = int((frac/2) * W)
    return img[dh:H-dh, dw:W-dw]

def avgpool2x2(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    x = img[:H2, :W2]
    if img.ndim == 2:
        return x.reshape(H2//2, 2, W2//2, 2).mean(axis=(1, 3)).astype(np.float32)
    else:
        # für 3-Kanal (falls nötig)
        return x.reshape(H2//2, 2, W2//2, 2, img.shape[2]).mean(axis=(1, 3)).astype(np.float32)
    
def downscale_times(img: np.ndarray, times: int = 5) -> np.ndarray:
    y = img
    for _ in range(times):
        y = avgpool2x2(y)
    return y

def edges_central_diff(img: np.ndarray) -> np.ndarray:
    # erwartet Graubild float32 [0,1]
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)

    gx[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
    gy[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])

    gx[:, 0]  = img[:, 1]  - img[:, 0]
    gx[:, -1] = img[:, -1] - img[:, -2]
    gy[0, :]  = img[1, :]  - img[0, :]
    gy[-1, :] = img[-1, :] - img[-2, :]

    mag = np.sqrt(gx*gx + gy*gy)
    # leicht normieren für robustes Matching
    mag -= mag.mean()
    std = mag.std() + 1e-12
    return (mag / std).astype(np.float32)

def convolve2d_valid(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    # sehr einfache 2D-Faltung "valid"; img und k sind 2D
    kh, kw = k.shape
    H, W = img.shape
    out = np.zeros((H - kh + 1, W - kw + 1), dtype=np.float32)
    # Flip für Faltung
    kf = np.flipud(np.fliplr(k)).astype(np.float32)
    for y in range(out.shape[0]):
        ys = y
        for x in range(out.shape[1]):
            xs = x
            patch = img[ys:ys+kh, xs:xs+kw]
            out[y, x] = np.sum(patch * kf)
    return out

def pad_reflect(img: np.ndarray, pad: int = 1) -> np.ndarray:
    # Randspiegelung für 3x3-Kerne
    return np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')

def edges_prewitt(img: np.ndarray) -> np.ndarray:
    # Prewitt-Kerne
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[ 1,  1,  1],
                   [ 0,  0,  0],
                   [-1, -1, -1]], dtype=np.float32)

    ip = pad_reflect(img, 1)
    gx = convolve2d_valid(ip, kx)
    gy = convolve2d_valid(ip, ky)
    mag = np.sqrt(gx*gx + gy*gy)

    mag -= mag.mean()
    std = mag.std() + 1e-12
    return (mag / std).astype(np.float32)

# ---------- Hilfsfunktion: kompletter 3-Schritt ----------
def prepare_for_alignment(gray_float_img: np.ndarray,
                          crop_frac: float = 0.10,
                          down_times: int = 5,
                          edge_method: str = 'central') -> np.ndarray:
    """
    gray_float_img: Graubild float32 in [0,1]
    edge_method: 'central' oder 'prewitt'
    """
    x = crop_borders(gray_float_img, crop_frac)
    x = downscale_times(x, down_times)
    if edge_method == 'prewitt':
        e = edges_prewitt(x)
    else:
        e = edges_central_diff(x)
    return e


def resize_half(img: np.ndarray) -> np.ndarray:
    return avgpool2x2(img)

def grad_mag(img: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (img[:, 2:] - img[:, :-2])
    gy[1:-1, :] = 0.5 * (img[2:, :] - img[:-2, :])
    gx[:, 0]  = img[:, 1]  - img[:, 0]
    gx[:, -1] = img[:, -1] - img[:, -2]
    gy[0, :]  = img[1, :]  - img[0, :]
    gy[-1, :] = img[-1, :] - img[-2, :]
    m = np.sqrt(gx*gx + gy*gy)
    m -= m.mean()
    s = m.std() + 1e-12
    return (m / s).astype(np.float32)

def build_pyr_to_100(img: np.ndarray, target_min: int = 100, max_levels: int = 6):
    pyr = [img]
    while min(img.shape) > target_min and len(pyr) < max_levels:
        img = resize_half(img)
        pyr.append(img)
    return pyr[::-1]  # coarse → fine

def align_pyramid_edges(
    ref: np.ndarray, tgt: np.ndarray,
    base_win: int = 20,
    border_frac: float = 0.12,    # fester Innenrand fürs Scoring
    metric: str = 'ncc',          # 'ncc' oder 'ssd'
    use_pc_init: bool = True,
    target_min: int = 120,        # ~100–120 für grobste Ebene
    max_levels: int = 6,
    edge_method: str = 'central'  # 'central' oder 'prewitt'
):
    # Pyramiden (Halbierung bis ~120px kleinste Kante)
    ref_pyr = build_pyr_to_100(ref, target_min=target_min, max_levels=max_levels)
    tgt_pyr = build_pyr_to_100(tgt, target_min=target_min, max_levels=max_levels)

    dy_total, dx_total = 0, 0

    def feature(x):
        return edges_prewitt(x) if edge_method == 'prewitt' else edges_central_diff(x)

    for lvl, (r, t) in enumerate(zip(ref_pyr, tgt_pyr)):
        # Feature-Bilder (volle Größe, kein Cropping!)
        r_feat = feature(r)
        t_feat = feature(t)

        # bisherige Schätzung hochskalieren
        if lvl > 0:
            dy_total *= 2
            dx_total *= 2

        # Initialschätzer: nur auf Level 0 und auf INTENSITÄTEN (stabiler als Edges)
        if use_pc_init and lvl == 0:
            H, W = r.shape
            inset0 = int(0.20 * min(H, W))
            r_int = r[inset0:H-inset0, inset0:W-inset0]
            t_int = t[inset0:H-inset0, inset0:W-inset0]
            dypc, dxpc = phase_correlation(r_int, t_int)
            dy_total += dypc
            dx_total += dxpc

        # Suchfenster pro Level
        if lvl == 0:
            win = 8
        elif lvl < len(ref_pyr) - 1:
            win = 4
        else:
            win = base_win

        # Border-Inset (nur fürs Scoring)
        Hf, Wf = r_feat.shape
        inset = int(border_frac * min(Hf, Wf) * 0.5)

        best = None
        for ddy in range(-win, win+1):
            for ddx in range(-win, win+1):
                dy = dy_total + ddy
                dx = dx_total + ddx

                # Overlap im vollen Koordinatensystem (keine Crops!)
                ov_r, ov_t = overlap_views(r_feat, t_feat, dy, dx)
                if ov_r is None:
                    continue

                # Innenrand abziehen (bleibt im gleichen Koordinatenrahmen)
                def inset_slice(slc, inset, max_len):
                    a, b = slc.start, slc.stop
                    a2 = max(a + inset, 0)
                    b2 = min(b - inset, max_len)
                    return None if b2 <= a2 else slice(a2, b2)

                yr = inset_slice(ov_r[0], inset, Hf); xr = inset_slice(ov_r[1], inset, Wf)
                yt = inset_slice(ov_t[0], inset, Hf); xt = inset_slice(ov_t[1], inset, Wf)
                if yr is None or xr is None or yt is None or xt is None:
                    continue

                A = r_feat[yr, xr]; B = t_feat[yt, xt]
                score = (ncc(A,B) if metric == 'ncc' else -ssd(A,B))

                # Tiebreaker: kleinere |dy|+|dx| bevorzugen
                key = (score, -(abs(dy) + abs(dx)))
                if best is None or key > best[0]:
                    best = (key, dy, dx)

        if best is not None:
            _, dy_total, dx_total = best

    return dy_total, dx_total
