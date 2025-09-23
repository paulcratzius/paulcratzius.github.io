import os
import numpy as np
from imageio.v2 import imread, imwrite
from scipy.signal import convolve2d
import cv2 as cv

def to_float01(img):
    arr = img.astype(np.float32)
    if np.issubdtype(img.dtype, np.integer):
        # scale by integer max (handles uint8→255, uint16→65535, etc.)
        maxv = np.iinfo(img.dtype).max
        arr = arr / float(maxv)
    else:
        # float input: assume already in [0,1], optionally clip
        arr = np.clip(arr, 0.0, 1.0)
    return arr

def to_uint8(img01):
    a = np.clip(img01, 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)

# Function to display arbitrary float32 arrays as images (needed for high frequency filters, as they can have negative values)
def minmax_vis(arr):
    a = arr.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-8:
        return np.zeros_like(a, np.uint8)
    return ((a - mn) / (mx - mn) * 255.0).astype(np.uint8)


def gaussian_kernel(ksize, sigma):
    g = cv.getGaussianKernel(ksize, sigma).astype(np.float32)  # (k,1)
    return (g @ g.T).astype(np.float32)                        # (k,k)


# applies a 2D convolution with 'same' size and zero padding. Applies Kernel K to each channel if input is 3D.
def conv_same(img_f32, K):
    """Convolve 2D or 3D image with kernel K channel-wise (same size, zero padding)."""
    if img_f32.ndim == 2:
        return convolve2d(img_f32, K, mode='same', boundary='fill', fillvalue=0.0).astype(np.float32)
    H, W, C = img_f32.shape
    out = np.empty_like(img_f32)
    for c in range(C):
        out[..., c] = convolve2d(img_f32[..., c], K, mode='same', boundary='fill', fillvalue=0.0).astype(np.float32)
    return out


# ---------- unsharp mask ----------
def unsharp_mask(img, ksize=9, sigma=1.6, alpha=1.0):
    """
    img in [0,1] (H,W[,C]), returns:
      blur, high (img - blur), sharp = clip(img + alpha*high)
    """
    K = gaussian_kernel(ksize, sigma)
    blur  = conv_same(img, K)
    high  = img - blur
    sharp = np.clip(img + alpha * high, 0.0, 1.0)
    return blur, high, sharp


# ---------- I/O runner ----------
def run_unsharp_for_images(
    images,
    out_dir="projects/project2/outputs/part2_unsharp",
    ksize=9,
    sigma_list=(1.0, 1.6, 2.0),
    alpha_list=(0.5, 1.0, 1.25),
):
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for name in images:
        in_path = f"projects/project2/images/{name}"
        stem = os.path.splitext(os.path.basename(name))[0]
        img = imread(in_path)
        img01 = to_float01(img)

        # Save original (as reference)
        imwrite(f"{out_dir}/{stem}_orig.jpg", to_uint8(img01))

        for sigma in sigma_list:
            for alpha in alpha_list:
                blur, high, sharp = unsharp_mask(img01, ksize=ksize, sigma=float(sigma), alpha=float(alpha))

                # Per-channel visualization for high (use magnitude across channels if RGB)
                if high.ndim == 3:
                    high_vis = np.linalg.norm(high, axis=2)  # simple way to show edges in one channel
                else:
                    high_vis = high

                imwrite(f"{out_dir}/{stem}_blur_s{sigma:.2f}.jpg",  to_uint8(blur))
                imwrite(f"{out_dir}/{stem}_high_s{sigma:.2f}.jpg",  minmax_vis(high_vis))
                imwrite(f"{out_dir}/{stem}_sharp_s{sigma:.2f}_a{alpha:.2f}.jpg", to_uint8(sharp))

                rows.append((stem, sigma, alpha))

    # Tiny CSV of what you produced (handy for wiring HTML if needed)
    with open(f"{out_dir}/params.csv", "w") as f:
        f.write("image,sigma,alpha\n")
        for (stem, sigma, alpha) in rows:
            f.write(f"{stem},{sigma},{alpha}\n")

if __name__ == "__main__":
    run_unsharp_for_images(
        images=["taj.jpg", "carrying-tobi.jpg", "half-dome.jpg"],
        ksize=9,
        sigma_list=(1.0, 1.6, 2.0),
        alpha_list=(0.5, 1.0, 1.25),
    )