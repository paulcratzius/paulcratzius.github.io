# projects/project2/code/part2-frequencies/hybrid_images.py

import os
import math
import numpy as np
from imageio.v2 import imread, imwrite
from scipy.signal import convolve2d
import cv2 as cv

# --- interaktives Alignment (rotiert immer das 1. Bild / im1) ---
from align_image_code import align_images  # erwartet float Bilder in [0,1]

# ================= Utils =================

def save_img(path, img01):
    """Robust speichern (uint8, korrekte Kanäle)."""
    u8 = to_uint8(img01)
    if u8.ndim == 3 and u8.shape[2] == 1:
        u8 = np.squeeze(u8, axis=2)  # JPEG "L"
    elif u8.ndim == 3 and u8.shape[2] > 3:
        u8 = u8[..., :3]
    imwrite(path, u8)

def to_float01(img):
    arr = img.astype(np.float32)
    if np.issubdtype(img.dtype, np.integer):
        maxv = np.iinfo(img.dtype).max
        arr = arr / float(maxv)
    else:
        arr = np.clip(arr, 0.0, 1.0)
    return arr

def to_uint8(img01):
    a = np.clip(img01, 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)

def grayscale(img01):
    """float32 Grau. Handhabt (H,W), (H,W,1), (H,W,3/4)."""
    if img01.ndim == 2:
        return img01.astype(np.float32)
    if img01.ndim == 3:
        c = img01.shape[2]
        if c == 1:
            return img01[..., 0].astype(np.float32)
        # RGBA -> ignoriere Alpha
        return (0.299*img01[...,0] + 0.587*img01[...,1] + 0.114*img01[...,2]).astype(np.float32)
    return np.squeeze(img01).astype(np.float32)

def minmax_vis(arr):
    a = arr.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-8: return np.zeros_like(a, np.uint8)
    return ((a - mn) / (mx - mn) * 255.0).astype(np.uint8)

def fft_log_image(img, dc_radius_frac=0.02, q_lo=2.0, q_hi=99.5, gamma=0.85):
    """
    Robuste FFT-Visualisierung (log |FFT|), skaliert über Quantile,
    ignoriert die DC-Nachbarschaft. Akzeptiert 2D, (H,W,1) und (H,W,3/4).
    """
    a = np.asarray(img, np.float32)
    if a.ndim == 3:
        if a.shape[2] == 1:
            a = a[..., 0]                # (H,W,1) -> (H,W)
        elif a.shape[2] >= 3:
            a = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32)

    # log-Magnitude
    F   = np.fft.fftshift(np.fft.fft2(a))
    mag = np.log1p(np.abs(F)).astype(np.float32)

    H, W = mag.shape
    cy, cx = H//2, W//2
    r = max(3, int(round(min(H, W) * dc_radius_frac)))
    Y, X = np.ogrid[:H, :W]
    dc_mask = (X - cx)**2 + (Y - cy)**2 <= r*r

    vals = mag[~dc_mask].ravel()
    lo = np.percentile(vals, q_lo)
    hi = np.percentile(vals, q_hi)
    if hi <= lo + 1e-12: hi = lo + 1.0

    z = (mag - lo) / (hi - lo)
    z = np.clip(z, 0.0, 1.0)
    if gamma != 1.0:
        z = np.power(z, gamma)

    return (z * 255.0 + 0.5).astype(np.uint8)




# ================= Gaussian / Convolution =================

def gaussian_kernel(ksize, sigma):
    g = cv.getGaussianKernel(ksize, sigma).astype(np.float32)  # (k,1)
    return (g @ g.T).astype(np.float32)                        # (k,k)

def conv_same(img_f32, K):
    if img_f32.ndim == 2:
        return convolve2d(img_f32, K, mode='same',
                          boundary='fill', fillvalue=0.0).astype(np.float32)
    H, W, C = img_f32.shape
    out = np.empty_like(img_f32)
    for c in range(C):
        out[..., c] = convolve2d(img_f32[..., c], K, mode='same',
                                 boundary='fill', fillvalue=0.0).astype(np.float32)
    return out

def ksize_for_sigma(sigma):
    # großzügig: ~6*sigma, dann auf ungerade runden
    return int(6*sigma + 1) | 1

# ======= Paper-gerecht: σ aus Cutoff (cycles/image) =======
# sigma_pixels ≈ 0.18739 * (N / f_c), N = min(H,W)
_CONST = math.sqrt(math.log(2)) / (math.sqrt(2)*math.pi)  # ≈ 0.187390625

def sigma_from_cutoff(shape_hw, f_c):
    H, W = shape_hw[:2]
    N = min(H, W)
    return max(0.5, _CONST * (N / max(1e-6, float(f_c))))

# ================= Filter & Hybrid =================

def low_pass(img01, sigma):
    K = gaussian_kernel(ksize_for_sigma(sigma), sigma)
    return conv_same(img01, K)

def high_pass_gray(img01, sigma):
    """High-pass als Grau (empfohlen im Paper), zero-mean."""
    g = grayscale(img01)
    low = low_pass(g, sigma)
    high = g - low
    high -= np.mean(high)
    return high

def make_hybrid(
    imgA, imgB,
    sigma_low, sigma_high,
    w_low=1.0, w_high=1.0,
    color_mode='low-color_high-gray',
    band_ratio=2.8,
    normalize_mode='to_low',
    high_to_low_ratio=0.75,
    high_target_rms=0.08,
    normalize_high=None  # ← nur zum Abfangen alter Aufrufe, wird ignoriert
):
    """
    imgA -> Low-Pass, imgB -> High-Pass.
    - DoG: highB = G(s) - G(s*band_ratio) (band-pass, weniger Schleier)
    - normalize_mode='to_low': skaliert highB-Energie relativ zur Energie von lowA
    """
    A = imgA.copy()
    B = imgB.copy()

    # Farbmodus
    if color_mode == 'gray':
        A = grayscale(A)[..., None]
        B = grayscale(B)[..., None]

    # Low-Pass (A)
    lowA = low_pass(A, sigma_low)

    # High-Pass (B) als Bandpass (DoG). Falls band_ratio≈1 -> normaler High-Pass
    if band_ratio and band_ratio > 1.0:
        g1 = low_pass(B, sigma_high)
        g2 = low_pass(B, sigma_high * band_ratio)
        highB = g1 - g2
    else:
        highB = B - low_pass(B, sigma_high)

    # Nur Luminanz für High-Pass, um Farb-Halos zu vermeiden
    if color_mode == 'low-color_high-gray' and A.ndim == 3:
        highB = grayscale(highB)[..., None]

    # Energie-Normierung des High-Pass
    # Zuerst mittelnull (damit 'std' wirklich Kantenenergie misst)
    highB = highB - np.mean(highB, axis=(0,1), keepdims=True)

    if normalize_mode == 'to_low':
        # std(highB) = high_to_low_ratio * std(lowA)
        low_std  = float(np.std(lowA))
        high_std = float(np.std(highB)) + 1e-12
        target   = high_to_low_ratio * low_std
        highB    = highB * (target / high_std)
    else:  # 'fixed'
        rms      = np.sqrt(np.mean(highB**2) + 1e-12)
        highB    = highB * (high_target_rms / rms)

    # Mischung
    hybrid = np.clip(w_low*lowA + w_high*highB, 0.0, 1.0)
    return lowA, highB, hybrid



# ================= Pyramids =================

def gaussian_pyramid(img01, levels=5, sigma=1.0):
    pyr = [img01]
    cur = img01
    for _ in range(1, levels):
        cur = low_pass(cur, sigma)
        cur = cur[::2, ::2, ...] if cur.ndim == 3 else cur[::2, ::2]
        pyr.append(cur)
    return pyr

def laplacian_pyramid(img01, levels=5, sigma=1.0):
    gp = gaussian_pyramid(img01, levels, sigma)
    lp = []
    for i in range(len(gp) - 1):
        up = cv.resize(gp[i+1], (gp[i].shape[1], gp[i].shape[0]),
                       interpolation=cv.INTER_LINEAR)
        # Kanal-Form compat
        if gp[i].ndim == 3 and up.ndim == 2: up = up[..., None]
        if gp[i].ndim == 2 and up.ndim == 3: up = np.squeeze(up, 2)
        up_blur = low_pass(up, sigma)
        if gp[i].ndim == 3 and up_blur.ndim == 2: up_blur = up_blur[..., None]
        if gp[i].ndim == 2 and up_blur.ndim == 3: up_blur = np.squeeze(up_blur, 2)
        lp.append(np.clip(gp[i] - up_blur, 0.0, 1.0))
    lp.append(gp[-1])
    return gp, lp

def save_pyramid_grid(levels, out_path, per_row=5):
    def to_vis_uint8(img):
        a = img.astype(np.float32)
        if a.ndim == 2: a = a[..., None]
        if a.shape[2] > 3: a = a[..., :3]
        mn, mx = float(np.min(a)), float(np.max(a))
        if mx - mn < 1e-8:
            a = np.zeros_like(a, dtype=np.uint8)
        else:
            a = (a - mn) / (mx - mn)
            a = (a * 255.0 + 0.5).astype(np.uint8)
        if a.shape[2] == 1: a = np.repeat(a, 3, axis=2)
        return a

    thumbs = [to_vis_uint8(L) for L in levels]
    if not thumbs: return

    rows, i = [], 0
    while i < len(thumbs):
        this_row = thumbs[i:i+per_row]
        target_h = min(img.shape[0] for img in this_row)
        resized = []
        for img in this_row:
            h, w = img.shape[:2]
            if h != target_h:
                scale = target_h / float(h)
                img = cv.resize(img, (max(1,int(round(w*scale))), target_h),
                                interpolation=cv.INTER_AREA)
            if img.ndim == 2: img = np.repeat(img[..., None], 3, axis=2)
            elif img.shape[2] == 1: img = np.repeat(img, 3, axis=2)
            resized.append(img)
        rows.append(np.hstack(resized))
        i += per_row

    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        h, w = r.shape[:2]
        if w < max_w:
            pad = np.full((h, max_w - w, 3), 255, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded_rows.append(r)
    grid = np.vstack(padded_rows)
    imwrite(out_path, grid)

# ================= Runner =================

def run_one_pair(
    imgA_path, imgB_path, pair_name,
    f_low_cpi=None,   # gewünschter Low-Pass cutoff in cycles/image
    f_high_cpi=None,  # gewünschter High-Pass cutoff in cycles/image
    sigma_low=None,   # alternativ: direkte Sigmas
    sigma_high=None,
    w_low=1.0, w_high=1.4,
    color_mode='low-color_high-gray',
    out_root="projects/project2/outputs/part2_hybrid",
    do_align=True,
    rotate_highpass=True,           # rotiere gezielt das High-Pass-Bild (B) zu A
    normalize_high=True             # High-Pass Kontrast leicht normalisieren
):
    """
    imgA_path -> Low-Pass (A)
    imgB_path -> High-Pass (B)
    """
    os.makedirs(os.path.join(out_root, pair_name), exist_ok=True)
    out_dir = os.path.join(out_root, pair_name)

    A0 = to_float01(imread(imgA_path))
    B0 = to_float01(imread(imgB_path))

    save_img(f"{out_dir}/A_original.jpg", A0)
    save_img(f"{out_dir}/B_original.jpg", B0)

    # ----- Alignment (wichtig: Prompt erklärt Klick-Reihenfolge) -----
    if do_align:
        if rotate_highpass:
            print(f"[{pair_name}] Aligning … ZUERST 2 Punkte auf B (High) klicken, DANN 2 Punkte auf A (Low). Danach Fenster schließen.")
            # align_images rotiert immer das erste Bild → wir geben B zuerst
            B_al, A_al = align_images(B0, A0)
            A, B = A_al, B_al
        else:
            print(f"[{pair_name}] Aligning … ZUERST 2 Punkte auf A (Low), DANN 2 Punkte auf B (High). Danach Fenster schließen.")
            A_al, B_al = align_images(A0, B0)
            A, B = A_al, B_al
    else:
        H = min(A0.shape[0], B0.shape[0]); W = min(A0.shape[1], B0.shape[1])
        A = A0[:H,:W,...]; B = B0[:H,:W,...]

    save_img(f"{out_dir}/A_aligned.jpg", A)
    save_img(f"{out_dir}/B_aligned.jpg", B)

    # ----- Cutoffs -> Sigmas (Paper) -----
    if sigma_low is None and f_low_cpi is not None:
        sigma_low  = sigma_from_cutoff(A.shape, f_low_cpi)
    if sigma_high is None and f_high_cpi is not None:
        sigma_high = sigma_from_cutoff(B.shape, f_high_cpi)
    if sigma_low is None or sigma_high is None:
        # Fallback, falls nichts übergeben wurde
        # (funktioniert, ist aber weniger „paper-treu“)
        sigma_low  = sigma_low  or 7.0
        sigma_high = sigma_high or 2.5

    # ----- Filter + Hybrid -----
    lowA, highB, hybrid = make_hybrid(
        A, B,
        sigma_low, sigma_high,
        w_low=w_low, w_high=w_high,
        color_mode=color_mode,
        band_ratio=2.4,          # 2.2–2.8 ok
        normalize_high=True,
        high_target_rms=0.10     # ggf. 0.08–0.12 ausprobieren
        # normalize_low bleibt False (Default)
    )



    # ----- Speichern (High-Vis als 2D) -----
    save_img(f"{out_dir}/lowA_sigma{sigma_low:.2f}.jpg", lowA)
    save_img(f"{out_dir}/highB_sigma{sigma_high:.2f}.jpg", minmax_vis(highB))
    save_img(f"{out_dir}/hybrid.jpg", hybrid)

    # ----- FFTs (auf Grau) -----
    A_gray = grayscale(A); B_gray = grayscale(B)
    lowA_gray   = grayscale(lowA)   if lowA.ndim==3   else lowA
    hybrid_gray = grayscale(hybrid) if hybrid.ndim==3 else hybrid

    save_img(f"{out_dir}/fft_A.jpg",      fft_log_image(A_gray))
    save_img(f"{out_dir}/fft_B.jpg",      fft_log_image(B_gray))
    save_img(f"{out_dir}/fft_lowA.jpg",   fft_log_image(lowA_gray))
    save_img(f"{out_dir}/fft_highB.jpg",  fft_log_image(highB))
    save_img(f"{out_dir}/fft_hybrid.jpg", fft_log_image(hybrid_gray))

    # ----- Pyramiden (für „bestes“ Beispiel) -----
    gp = gaussian_pyramid(hybrid, levels=5, sigma=1.0)
    save_pyramid_grid(gp, f"{out_dir}/gaussian_pyramid.jpg", per_row=5)
    _, lp = laplacian_pyramid(hybrid, levels=5, sigma=1.0)
    save_pyramid_grid(lp, f"{out_dir}/laplacian_pyramid.jpg", per_row=5)

    # ----- Log -----
    with open(f"{out_dir}/params.txt","w") as f:
        f.write(f"pair={pair_name}\n")
        f.write(f"cutoff_low_cpi={f_low_cpi}, cutoff_high_cpi={f_high_cpi}\n")
        f.write(f"sigma_low={sigma_low:.4f}, sigma_high={sigma_high:.4f}\n")
        f.write(f"w_low={w_low}, w_high={w_high}\n")
        f.write(f"color_mode={color_mode}\n")
        f.write(f"rotate_highpass={rotate_highpass}\n")
        f.write(f"aligned={do_align}\n")

    print(f"[{pair_name}] DONE → {out_dir}")

if __name__ == "__main__":
    IMG_ROOT = "projects/project2/images"

    # Empfehlung nach Paper:
    # - Low (z.B. Derek): starker Blur → niedrige cutoff (z.B. 6–8 cycles/image)
    # - High (z.B. Nutmeg): höhere cutoff (z.B. 18–24 cycles/image), High-Pass stärker gewichten
    pairs = [
        (f"{IMG_ROOT}/DerekPicture.jpg", f"{IMG_ROOT}/nutmeg.jpg",
        "derek_nutmeg",
        12.0,   # sigma_low ↑ → A glatter (aus der Nähe unauffälliger)
        2.6,   # sigma_high leicht ↑ → B verliert Mittelfrequenzen
        1.00,  # w_low moderat
        1.20,  # w_high runter → B weniger präsent
        "gray",
        True),

        # Hofmann (Low) + Berger (High)
        (f"{IMG_ROOT}/thomas_hofmann.jpeg", f"{IMG_ROOT}/albert_berger.jpeg",
        "hofmann_berger",
        8., 6.5, 0.90, 0.95, "low-color_high-gray", True),

        # Du + Ruben (High sollte stark sein)
        (f"{IMG_ROOT}/i-house-pic.JPG", f"{IMG_ROOT}/ruben.jpg",
        "house_ruben",
        9.0,  2.2,  0.55, 1.85, "low-color_high-gray", True),
    ]



    for (A, B, name, s_low, s_high, wl, wh, mode, rotB) in pairs:
        run_one_pair(
            A, B, name,
            sigma_low=s_low, sigma_high=s_high,
            w_low=wl, w_high=wh,
            color_mode=mode,
            do_align=True,
            rotate_highpass=rotB
        )

