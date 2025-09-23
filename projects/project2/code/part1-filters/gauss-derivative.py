import cv2 as cv
import numpy as np
from scipy.signal import convolve2d
import os
from imageio.v2 import imread, imwrite

# ----- Kernels -----
Dx = np.array([[1, 0, -1]], dtype=np.float64)     # (1,3)
Dy = np.array([[1],[0],[-1]], dtype=np.float64)   # (3,1)

# ----- Helpers -----
def ensure_gray_f64(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    return image.astype(np.float64)

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    g = cv.getGaussianKernel(size, sigma).astype(np.float64)  # (k,1)
    return (g @ g.T).astype(np.float64)

def to_u8(a: np.ndarray) -> np.ndarray:
    """Independent min-max (use only for filters/blur previews)."""
    a = a.astype(np.float64)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-12:
        return np.zeros_like(a, np.uint8)
    return np.clip((a - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)

def to_u8_pair(A: np.ndarray, B: np.ndarray, robust: bool = True, p_lo=1.0, p_hi=99.0):
    """Normalize A and B to the SAME 0–255 scale (crucial for fair visual comparison)."""
    A = A.astype(np.float64); B = B.astype(np.float64)
    both = np.concatenate([A.ravel(), B.ravel()])
    if robust:
        vmin = float(np.percentile(both, p_lo))
        vmax = float(np.percentile(both, p_hi))
    else:
        vmin, vmax = float(both.min()), float(both.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(A, np.uint8), np.zeros_like(B, np.uint8)
    Au8 = np.clip((A - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    Bu8 = np.clip((B - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    return Au8, Bu8

def save_diff_u8(path: str, A: np.ndarray, B: np.ndarray, pct: float = 99.5):
    """Save |A-B| with a robust fixed scale so small differences are visible."""
    D = np.abs(A.astype(np.float64) - B.astype(np.float64))
    vmax = float(np.percentile(D, pct))
    if vmax < 1e-12: vmax = 1.0
    imwrite(path, np.clip(D / vmax * 255.0, 0, 255).astype(np.uint8))

# ----- Pipelines -----
def create_blurred_image(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Blur image with Gaussian kernel (same size output, zero padding)."""
    img = ensure_gray_f64(image)
    K = gaussian_kernel(kernel_size, sigma)
    blurred = convolve2d(img, K, mode='same', boundary='fill', fillvalue=0.0)
    return blurred.astype(np.float64)

def two_step_gradient(image: np.ndarray, kernel_size: int, sigma: float):
    """
    Blur with Gaussian, then compute Dx, Dy and magnitude.
    Returns: (Gx_blur, Gy_blur, mag_blur, blurred_image)
    """
    blurred = create_blurred_image(image, kernel_size, sigma)
    Gx = convolve2d(blurred, Dx, mode='same', boundary='fill', fillvalue=0.0)
    Gy = convolve2d(blurred, Dy, mode='same', boundary='fill', fillvalue=0.0)
    mag = np.sqrt(Gx**2 + Gy**2)
    return Gx.astype(np.float64), Gy.astype(np.float64), mag.astype(np.float64), blurred

def gaussian_derivative_filters(kernel_size: int, sigma: float):
    """
    Build 2D DoG filters (k×k) via separability:
      DoGx(y,x) = g(y) * (g * [1,0,-1])(x)
      DoGy(y,x) = (g * [1,0,-1])(y) * g(x)
    Returns: (DoGx, DoGy) as float64.
    """
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    g = cv.getGaussianKernel(kernel_size, sigma).astype(np.float64).ravel()  # (k,)
    diff1d = np.array([1, 0, -1], dtype=np.float64)  # 1D central difference
    dog1d = np.convolve(g, diff1d, mode='same')      # (k,)

    DoGx = np.outer(g,     dog1d).astype(np.float64)  # Gaussian in y, derivative in x
    DoGy = np.outer(dog1d, g    ).astype(np.float64)  # derivative in y, Gaussian in x
    return DoGx, DoGy

def gaussian_derivative_filters_2d(kernel_size: int, sigma: float):
    """
    Build DoGx/DoGy by convolving the 2D Gaussian G (k×k) with Dx/Dy in 'full',
    then center-crop each result back to (k×k). This matches (G * Dx), (G * Dy).
    """
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    G2D = gaussian_kernel(kernel_size, sigma)  # (k,k)

    # X-derivative
    DoGx_full = convolve2d(G2D, Dx, mode='full', boundary='fill', fillvalue=0.0)
    pad_y_x = (DoGx_full.shape[0] - kernel_size) // 2   # 0
    pad_x_x = (DoGx_full.shape[1] - kernel_size) // 2   # 1
    DoGx = DoGx_full[pad_y_x:pad_y_x + kernel_size, pad_x_x:pad_x_x + kernel_size].astype(np.float64)

    # Y-derivative
    DoGy_full = convolve2d(G2D, Dy, mode='full', boundary='fill', fillvalue=0.0)
    pad_y_y = (DoGy_full.shape[0] - kernel_size) // 2   # 1
    pad_x_y = (DoGy_full.shape[1] - kernel_size) // 2   # 0
    DoGy = DoGy_full[pad_y_y:pad_y_y + kernel_size, pad_x_y:pad_x_y + kernel_size].astype(np.float64)

    return DoGx, DoGy

# ----- Runner -----
def run_part13(cameraman_path,
               out_dir="projects/project2/outputs/part1_filters/dog",
               ksize=9, sigma=1.6):
    """
    Part 1.3 runner:
      - uses two_step_gradient() for Blur → Dx/Dy/Mag
      - builds DoGx/DoGy and runs one-shot DoG
      - saves visualizations with SHARED scaling for fair comparison
      - writes CSV with global & inner-region diffs
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load → gray/float64
    img = ensure_gray_f64(imread(cameraman_path))

    # (A) Blur → finite differences
    Gx_b, Gy_b, M_b, blur = two_step_gradient(img, ksize, sigma)

    # (B) One-shot DoG
    DoGx, DoGy = gaussian_derivative_filters_2d(ksize, sigma)
    Gx_d = convolve2d(img, DoGx, mode='same', boundary='fill', fillvalue=0.0)
    Gy_d = convolve2d(img, DoGy, mode='same', boundary='fill', fillvalue=0.0)
    M_d  = np.sqrt(Gx_d**2 + Gy_d**2)

    # Sanity: separable vs 2D build for filters
    DoGx_sep, DoGy_sep = gaussian_derivative_filters(ksize, sigma)
    print("max|DoGx_sep - DoGx_ref| =", float(np.max(np.abs(DoGx_sep - DoGx))))
    print("max|DoGy_sep - DoGy_ref| =", float(np.max(np.abs(DoGy_sep - DoGy))))

    # ----- Save images -----
    # Blur preview + filters (independent scaling is fine here)
    imwrite(f"{out_dir}/cameraman_blur.jpg", to_u8(blur))
    imwrite(f"{out_dir}/dogx_filter.jpg",    to_u8(DoGx))
    imwrite(f"{out_dir}/dogy_filter.jpg",    to_u8(DoGy))

    # Derivative pairs with the SAME scaling (critical)
    Gx_b_u8, Gx_d_u8 = to_u8_pair(Gx_b, Gx_d, robust=True)
    Gy_b_u8, Gy_d_u8 = to_u8_pair(Gy_b, Gy_d, robust=True)
    Mb_u8,   Md_u8   = to_u8_pair(M_b,  M_d,  robust=True)

    imwrite(f"{out_dir}/cameraman_blur_dx.jpg",  Gx_b_u8)
    imwrite(f"{out_dir}/cameraman_dog_dx.jpg",   Gx_d_u8)
    imwrite(f"{out_dir}/cameraman_blur_dy.jpg",  Gy_b_u8)
    imwrite(f"{out_dir}/cameraman_dog_dy.jpg",   Gy_d_u8)
    imwrite(f"{out_dir}/cameraman_blur_mag.jpg", Mb_u8)
    imwrite(f"{out_dir}/cameraman_dog_mag.jpg",  Md_u8)

    # Difference maps (robust fixed scale)
    save_diff_u8(f"{out_dir}/diff_dx.jpg",  Gx_b, Gx_d)
    save_diff_u8(f"{out_dir}/diff_dy.jpg",  Gy_b, Gy_d)
    save_diff_u8(f"{out_dir}/diff_mag.jpg", M_b,  M_d)

    # ----- Numeric diffs -----
    # Global
    diff_x = float(np.max(np.abs(Gx_b - Gx_d)))
    diff_y = float(np.max(np.abs(Gy_b - Gy_d)))
    diff_m = float(np.max(np.abs(M_b  - M_d)))

    # Inner region (remove border where padding paths differ)
    def inner_diff_stats(Gx_b, Gy_b, Gx_d, Gy_d, k):
        H, W = Gx_b.shape
        my_x, mx_x = k//2, k//2 + 1        # safe margins for Gx
        my_y, mx_y = k//2 + 1, k//2        # safe margins for Gy
        ix = (slice(my_x, H - my_x), slice(mx_x, W - mx_x))
        iy = (slice(my_y, H - my_y), slice(mx_y, W - mx_y))
        dx_inner = float(np.max(np.abs(Gx_b[ix] - Gx_d[ix])))
        dy_inner = float(np.max(np.abs(Gy_b[iy] - Gy_d[iy])))
        iyx = (slice(max(my_x, my_y), H - max(my_x, my_y)),
               slice(max(mx_x, mx_y), W - max(mx_x, mx_y)))
        Mb = np.sqrt(Gx_b**2 + Gy_b**2)
        Md = np.sqrt(Gx_d**2 + Gy_d**2)
        mag_inner = float(np.max(np.abs(Mb[iyx] - Md[iyx])))
        return dx_inner, dy_inner, mag_inner

    dx_in, dy_in, m_in = inner_diff_stats(Gx_b, Gy_b, Gx_d, Gy_d, ksize)
    print("Inner diffs — max|dx|:", dx_in, " max|dy|:", dy_in, " max|mag|:", m_in)

    # CSV
    with open(f"{out_dir}/results.csv", "w") as f:
        f.write("ksize,sigma,max_abs_diff_x,max_abs_diff_y,max_abs_diff_mag,inner_max_diff_x,inner_max_diff_y,inner_max_diff_mag\n")
        f.write(f"{ksize},{sigma},{diff_x:.8f},{diff_y:.8f},{diff_m:.8f},{dx_in:.8f},{dy_in:.8f},{m_in:.8f}\n")

# ----- Main -----
if __name__ == "__main__":
    run_part13("projects/project2/images/cameraman.png")
