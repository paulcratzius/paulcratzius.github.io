import numpy as np
from scipy.signal import convolve2d
import os
from imageio.v2 import imread, imwrite
from time import perf_counter

# ========= Kernels =========
Dx = np.array([[1, 0, -1]], dtype=np.float32)     # (1,3)
Dy = np.array([[1],[0],[-1]], dtype=np.float32)   # (3,1)

# ========= Helpers =========
def setup(image, kernel):
    """Return (padded_image, flipped_kernel) with zero padding. Grayscale + float32 inside."""
    if image.ndim == 3:
        image = np.mean(image, axis=2).astype(np.float32)
    else:
        image = image.astype(np.float32)

    ph = kernel.shape[0] // 2
    pw = kernel.shape[1] // 2
    padded_image = np.pad(image, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
    flipped_kernel = np.flipud(np.fliplr(kernel))
    return padded_image, flipped_kernel

def convolve_4(image, kernel):
    """4 nested loops, 'same' size, zero padding."""
    if image.ndim == 3:
        image = np.mean(image, axis=2).astype(np.float32)
    else:
        image = image.astype(np.float32)
    padded, flipped = setup(image, kernel)
    H, W = image.shape
    kH, kW = flipped.shape
    out = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            acc = 0.0
            for m in range(kH):
                for n in range(kW):
                    acc += padded[i + m, j + n] * flipped[m, n]
            out[i, j] = acc
    return out

def convolve_2(image, kernel):
    """2 loops + vectorized inner product, 'same' size, zero padding."""
    if image.ndim == 3:
        image = np.mean(image, axis=2).astype(np.float32)
    else:
        image = image.astype(np.float32)
    padded, flipped = setup(image, kernel)
    H, W = image.shape
    kH, kW = flipped.shape
    out = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            region = padded[i:i+kH, j:j+kW]
            out[i, j] = np.sum(region * flipped)
    return out

def convolve_scipy(image, kernel):
    """SciPy convolve2d, 'same', zero boundary."""
    if image.ndim == 3:
        image = np.mean(image, axis=2).astype(np.float32)
    else:
        image = image.astype(np.float32)
    # SciPy macht echtes Convolution → Kernel NICHT selbst flippen.
    y = convolve2d(image, kernel.astype(np.float32), mode='same', boundary='fill', fillvalue=0.0)
    return y.astype(np.float32)

def _to_gray_float(img):
    return np.mean(img, axis=2).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)

def _save_uint8_linear(img_f32, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imwrite(path, np.clip(img_f32, 0, 255).astype(np.uint8))

def _save_uint8_rescaled(img_f32, path):
    """For Dx/Dy: high-contrast representation through min-max normalization to [0,255]."""
    mn, mx = float(np.min(img_f32)), float(np.max(img_f32))
    if mx - mn < 1e-8:
        vis = np.zeros_like(img_f32, dtype=np.uint8)
    else:
        vis = ( (img_f32 - mn) / (mx - mn) * 255.0 ).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imwrite(path, vis)

def _record(rows, stem, kernel_name, method, ref, y, t, atol=1e-5):
    max_abs_err = float(np.max(np.abs(y - ref)))
    allc = bool(np.allclose(y, ref, atol=atol))
    rows.append((stem, kernel_name, method, t, max_abs_err, int(allc)))

def run_part11(image_path,
               out_dir="projects/project2/outputs/part1_filters/conv_examples",
               out_stem="i-house-pic"):
    """
    Computes Box9, Dx, Dy with 4-Loops / 2-Loops / SciPy.
    Saves images and writes a CSV with times & accuracy (against SciPy per kernel).
    """
    os.makedirs(out_dir, exist_ok=True)
    img = imread(image_path)
    gray = _to_gray_float(img)

    # Input image
    _save_uint8_linear(gray, os.path.join(out_dir, f"{out_stem}_input.jpg"))

    # Kernel set
    kernels = {
        "box9": (np.ones((9, 9), np.float32) / 81.0),
        "Dx": Dx,
        "Dy": Dy,
    }

    rows = []  # CSV rows

    for kname, K in kernels.items():
        # --- Reference (SciPy) ---
        t0 = perf_counter()
        ref = convolve_scipy(gray, K)
        t_scipy = perf_counter() - t0

        # Saving (Visualization: Box linear; Dx/Dy rescaled)
        if kname == "box9":
            _save_uint8_linear(ref, os.path.join(out_dir, f"{out_stem}_{kname}_scipy.jpg"))
        else:
            _save_uint8_rescaled(ref, os.path.join(out_dir, f"{out_stem}_{kname}_scipy.jpg"))

        _record(rows, out_stem, kname, "scipy", ref, ref, t_scipy)

        # --- 4-Loops ---
        t0 = perf_counter()
        y4 = convolve_4(gray, K)
        t4 = perf_counter() - t0
        if kname == "box9":
            _save_uint8_linear(y4, os.path.join(out_dir, f"{out_stem}_{kname}_4loops.jpg"))
        else:
            _save_uint8_rescaled(y4, os.path.join(out_dir, f"{out_stem}_{kname}_4loops.jpg"))
        _record(rows, out_stem, kname, "4loops", ref, y4, t4)

        # --- 2-Loops ---
        t0 = perf_counter()
        y2 = convolve_2(gray, K)
        t2 = perf_counter() - t0
        if kname == "box9":
            _save_uint8_linear(y2, os.path.join(out_dir, f"{out_stem}_{kname}_2loops.jpg"))
        else:
            _save_uint8_rescaled(y2, os.path.join(out_dir, f"{out_stem}_{kname}_2loops.jpg"))
        _record(rows, out_stem, kname, "2loops", ref, y2, t2)

    # --- CSV ---
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w") as f:
        f.write("image,kernel,method,time_sec,max_abs_err_vs_scipy,allclose\n")
        for (imgname, kname, method, t, err, allc) in rows:
            f.write(f"{imgname},{kname},{method},{t:.6f},{err:.8f},{allc}\n")
    return csv_path

if __name__ == "__main__":
    run_part11("projects/project2/images/i-house-pic.jpg")
