import numpy as np
from scipy.signal import convolve2d
import os
from imageio.v2 import imread, imwrite

Dx = np.array([[1, 0, -1]], dtype=np.float32)     # (1,3)
Dy = np.array([[1],[0],[-1]], dtype=np.float32)   # (3,1)

def partial_derivative_x(image: np.ndarray) -> np.ndarray:
    """∂/∂x via 1D finite difference Dx; same-size, zero padding."""
    return convolve2d(image.astype(np.float32), Dx, mode='same',
                      boundary='fill', fillvalue=0.0).astype(np.float32)

def partial_derivative_y(image: np.ndarray) -> np.ndarray:
    """∂/∂y via 1D finite difference Dy; same-size, zero padding."""
    return convolve2d(image.astype(np.float32), Dy, mode='same',
                      boundary='fill', fillvalue=0.0).astype(np.float32)

def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """||∇I|| = sqrt(Gx^2 + Gy^2)."""
    dx = partial_derivative_x(image)
    dy = partial_derivative_y(image)
    return np.sqrt(dx**2 + dy**2, dtype=np.float32)

def to_uint8_minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)

def run_part12(
    cameraman_path: str,
    out_dir: str = "projects/project2/outputs/part1_filters/finite_diff",
    threshold_mode: str = "percentile",
    threshold_value: float = 89,
):
    """Compute Dx, Dy, magnitude, and a binary edge map for the cameraman image and save them."""
    os.makedirs(out_dir, exist_ok=True)

    img = imread(cameraman_path)
    img = (np.mean(img, axis=2) if img.ndim == 3 else img).astype(np.float32)

    # derivatives
    Gx = partial_derivative_x(img)
    Gy = partial_derivative_y(img)
    mag = np.sqrt(Gx**2 + Gy**2, dtype=np.float32)

    # visualizations
    imwrite(os.path.join(out_dir, "cameraman_dx.jpg"),  to_uint8_minmax(Gx))
    imwrite(os.path.join(out_dir, "cameraman_dy.jpg"),  to_uint8_minmax(Gy))
    imwrite(os.path.join(out_dir, "cameraman_mag.jpg"), to_uint8_minmax(mag))

    # threshold selection
    if threshold_mode == "percentile":
        t = np.percentile(mag, float(threshold_value))
    elif threshold_mode == "mean_std":
        k = float(threshold_value)  # e.g., 1.0
        t = float(mag.mean() + k * mag.std())
    else:
        t = float(threshold_value)

    edges = (mag >= t).astype(np.uint8) * 255
    imwrite(os.path.join(out_dir, "cameraman_edges.jpg"), edges)

    return {"threshold": t, "mode": threshold_mode}

if __name__ == "__main__":
    # Keep paths consistent with your website:
    CAM_PATH = "projects/project2/images/cameraman.png"
    info = run_part12(CAM_PATH)
    print(f"[Part 1.2] Saved outputs. Threshold={info['threshold']:.4f} ({info['mode']})")
