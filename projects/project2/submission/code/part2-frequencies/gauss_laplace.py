import os
import numpy as np
from imageio.v2 import imread, imwrite
import cv2 as cv

# -------------------- small utils --------------------
def to_float01(img):
    a = img.astype(np.float32)
    if np.issubdtype(img.dtype, np.integer):
        a /= np.iinfo(img.dtype).max
    return np.clip(a, 0.0, 1.0)

def to_uint8(img01):
    return np.clip(img01 * 255.0 + 0.5, 0, 255).astype(np.uint8)

def save_img(path, img01):
    x = to_uint8(img01)
    if x.ndim == 3 and x.shape[2] == 1: x = x[..., 0]
    if x.ndim == 3 and x.shape[2] > 3:  x = x[..., :3]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imwrite(path, x)

def gaussian_kernel(ksize, sigma):
    g = cv.getGaussianKernel(ksize, sigma).astype(np.float32)  # (k,1)
    return (g @ g.T).astype(np.float32)                        # (k,k)

def conv_same(img_f32, K):
    if img_f32.ndim == 2:
        return cv.filter2D(img_f32, -1, K, borderType=cv.BORDER_CONSTANT)
    out = np.empty_like(img_f32)
    for c in range(img_f32.shape[2]):
        out[..., c] = cv.filter2D(img_f32[..., c], -1, K, borderType=cv.BORDER_CONSTANT)
    return out

def gaussian_blur(img01, sigma, ksize=None):
    if ksize is None:
        ksize = int(6*sigma + 1) | 1
    return conv_same(img01, gaussian_kernel(ksize, sigma))

def vis01(x):
    """Per-image min-max for visualization."""
    a = x.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-8:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

# -------------------- stacks (NO DOWNSAMPLING) --------------------
def gaussian_stack(img01, levels=5, sigma=2.0):
    """[G0, G1, ..., G{L-1}] same size; each level is blurred again."""
    G = [img01.astype(np.float32)]
    cur = img01.astype(np.float32)
    for _ in range(1, levels):
        cur = gaussian_blur(cur, sigma)
        G.append(cur)
    return G

def laplacian_stack(img01, levels=5, sigma=2.0):
    """L_k = G_k - G_{k+1}; L_{L-1} = G_{L-1}."""
    G = gaussian_stack(img01, levels, sigma)
    L = []
    for k in range(levels - 1):
        L.append(G[k] - G[k+1])
    L.append(G[-1])
    return G, L

def mask_stack(mask01, levels=5, sigma=2.0):
    """Gaussian stack of mask in [0,1], single channel kept."""
    m = mask01[..., :1] if mask01.ndim == 3 else mask01[..., None]
    return gaussian_stack(m, levels, sigma)

# -------------------- multiresolution blend --------------------
def multires_blend(A01, B01, M01, levels=5, sigma=2.0):
    """
    A,B in [0,1] (same size, 3ch or 1ch). M in [0,1] (1→A, 0→B).
    Laplacian stacks for images; Gaussian stack for mask.
    """
    GA, LA = laplacian_stack(A01, levels, sigma)   # GA returned but not used directly
    GB, LB = laplacian_stack(B01, levels, sigma)
    GM = mask_stack(M01, levels, sigma)            # each GM[k] is HxWx1

    # level-wise blend
    blended_levels = []
    for k in range(levels):
        Mk = GM[k]
        LAk = LA[k] if LA[k].ndim == 3 else LA[k][..., None]
        LBk = LB[k] if LB[k].ndim == 3 else LB[k][..., None]
        blended_levels.append(Mk * LAk + (1.0 - Mk) * LBk)

    # reconstruction (no downsampling → sum)
    out = np.zeros_like(blended_levels[0], dtype=np.float32)
    for Bk in blended_levels:
        out += Bk
    out = np.clip(out, 0.0, 1.0)
    if out.shape[2] == 1: out = out[..., 0]
    return out, (LA, LB, GM), blended_levels

# -------------------- masks --------------------
def soft_vertical_mask(h, w, center=0.5, width=0.12, left_is_A=True):
    """Smooth vertical step. left_is_A: True→A on left; False→A on right."""
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    s = 1.0 / (1.0 + np.exp(-(x - center) / (width + 1e-8)))   # 0→1 left→right
    M = (1.0 - s) if left_is_A else s
    return np.repeat(M, h, axis=0)[..., None]

# -------------------- helpers --------------------
def center_crop_to_match(A, B):
    """
    Center-crop both images to the same (min) size so the main subject
    remains centered instead of being chopped from the top-left.
    """
    H = min(A.shape[0], B.shape[0])
    W = min(A.shape[1], B.shape[1])

    def ccrop(x):
        h, w = x.shape[:2]
        top  = max(0, (h - H) // 2)
        left = max(0, (w - W) // 2)
        if x.ndim == 3:
            return x[top:top+H, left:left+W, :]
        else:
            return x[top:top+H, left:left+W]

    return ccrop(A), ccrop(B)

# ---------- mountain-centric crop & triangle mask ----------

def _estimate_peak_y(img, search_top_rel=0.60):
    """
    Very simple peak detector for the snowy cone:
    - convert to gray
    - search only the top `search_top_rel` fraction (bright snow area)
    - return y of the brightest pixel
    """
    g = img.astype(np.float32)
    if g.ndim == 3: g = 0.299*g[...,0] + 0.587*g[...,1] + 0.114*g[...,2]
    h, w = g.shape[:2]
    h_search = max(1, int(h * search_top_rel))
    yx = np.unravel_index(np.argmax(g[:h_search, :], axis=None), (h_search, w))
    y_peak = int(yx[0])
    return y_peak

def crop_match_mountain_center(A, B, place_rel_y=0.30, search_top_rel=0.60):
    """
    Center-crop both images to the same size, but choose A's vertical crop so
    the detected mountain peak ends up at `place_rel_y` of the final crop.
    Horizontal cropping is always centered (to keep the cone in the middle).
    """
    # choose common output size
    H = min(A.shape[0], B.shape[0])
    W = min(A.shape[1], B.shape[1])

    # --- crop A (mountain) with peak control ---
    y_peak = _estimate_peak_y(A, search_top_rel=search_top_rel)
    # where the peak should land in the cropped image:
    y_target = int(place_rel_y * H)
    top_A = np.clip(y_peak - y_target, 0, max(0, A.shape[0] - H))
    left_A = max(0, (A.shape[1] - W) // 2)

    if A.ndim == 3:
        A_crop = A[top_A:top_A+H, left_A:left_A+W, :]
    else:
        A_crop = A[top_A:top_A+H, left_A:left_A+W]

    # --- crop B (skyline): plain centered so buildings stay centered ---
    top_B  = max(0, (B.shape[0] - H) // 2)
    left_B = max(0, (B.shape[1] - W) // 2)
    if B.ndim == 3:
        B_crop = B[top_B:top_B+H, left_B:left_B+W, :]
    else:
        B_crop = B[top_B:top_B+H, left_B:left_B+W]

    return A_crop, B_crop

def triangle_mask(h, w, apex_rel_y=0.22, base_rel_y=0.86, base_half_rel=0.38,
                  feather_px=60, A_inside=True):
    """
    Centered triangular mask:
    - apex at (w/2, apex_rel_y*h)
    - base along y = base_rel_y*h with width = 2*base_half_rel*w
    Everything inside triangle -> 1 (for A) unless A_inside=False.
    """
    cy = int(apex_rel_y * h)
    by = int(base_rel_y * h)
    half = int(base_half_rel * w)
    cx = w // 2

    pts = np.array([[cx, cy], [cx - half, by], [cx + half, by]], dtype=np.int32).reshape(-1,1,2)
    M = np.zeros((h, w), dtype=np.uint8)
    cv.fillPoly(M, [pts], 255)
    M = cv.GaussianBlur(M, (0,0), feather_px).astype(np.float32) / 255.0
    M = M[..., None]
    return M if A_inside else (1.0 - M)




# -------------------- Oraple (Figure 3.42) --------------------
def run_oraple(
    apple_path, orange_path,
    out_dir="projects/project2/outputs/part2_stacks/oraple",
    levels=5, sigma=2.0,
    mask_center=0.5, mask_width=0.12
):
    os.makedirs(out_dir, exist_ok=True)

    A = to_float01(imread(apple_path))   # apple (left)
    B = to_float01(imread(orange_path))  # orange (right)
    A, B = center_crop_to_match(A, B)

    H, W = A.shape[:2]
    M = soft_vertical_mask(H, W, mask_center, mask_width, left_is_A=True)

    # stacks
    GA, LA = laplacian_stack(A, levels, sigma)
    GB, LB = laplacian_stack(B, levels, sigma)
    GM = mask_stack(M, levels, sigma)

    # blend (final)
    blended, (LA2, LB2, GM2), blended_levels = multires_blend(A, B, M, levels, sigma)

    # --- save per-levels like Szelski (pick levels 0,2,4 as high/med/low) ---
    pick = [0, min(2, levels-2), min(4, levels-1)]
    tags = ['high','medium','low']
    for t, k in zip(tags, pick):
        save_img(f"{out_dir}/(a)_{t}_apple_L{k}.jpg",  vis01(LA[k]))
        save_img(f"{out_dir}/(b)_{t}_orange_L{k}.jpg", vis01(LB[k]))
        save_img(f"{out_dir}/(c)_{t}_avg_L{k}.jpg",    vis01(0.5*LA[k] + 0.5*LB[k]))

    # Weighted contributions (sum over levels with mask weights)
    contrib_apple  = np.zeros_like(A, dtype=np.float32)
    contrib_orange = np.zeros_like(B, dtype=np.float32)
    for k in range(levels-1):
        Mk = GM[k]
        LAk = LA[k] if LA[k].ndim == 3 else LA[k][..., None]
        LBk = LB[k] if LB[k].ndim == 3 else LB[k][..., None]
        contrib_apple  += Mk * LAk
        contrib_orange += (1.0 - Mk) * LBk
    # add low-pass bases
    contrib_apple  += GM[-1] * (GA[-1] if GA[-1].ndim==3 else GA[-1][...,None])
    contrib_orange += (1.0 - GM[-1]) * (GB[-1] if GB[-1].ndim==3 else GB[-1][...,None])

    save_img(f"{out_dir}/(j)_apple_weighted.jpg",  vis01(contrib_apple))
    save_img(f"{out_dir}/(k)_orange_weighted.jpg", vis01(contrib_orange))
    save_img(f"{out_dir}/(l)_oraple.jpg", blended)

    # Also dump full stacks for inspection
    for i in range(levels):
        save_img(f"{out_dir}/GA_{i}.jpg", vis01(GA[i]))
        save_img(f"{out_dir}/GB_{i}.jpg", vis01(GB[i]))
        save_img(f"{out_dir}/LA_{i}.jpg", vis01(LA[i]))
        save_img(f"{out_dir}/LB_{i}.jpg", vis01(LB[i]))
        save_img(f"{out_dir}/GM_{i}.jpg", vis01(GM[i]))

    # Inputs & mask
    save_img(f"{out_dir}/apple_input.jpg",  A)
    save_img(f"{out_dir}/orange_input.jpg", B)
    save_img(f"{out_dir}/mask.jpg", vis01(M))
    print(f"[oraple] saved to: {out_dir}")

# -------------------- generic custom blends --------------------
def run_custom_blend(
    A_path, B_path, out_dir,
    levels=5, sigma=2.0,
    mask_kind="vertical",           # 'vertical' | 'triangle'
    mask_kwargs=None,
    preprocess=None                 # None | 'mountain_center'
):
    os.makedirs(out_dir, exist_ok=True)
    mask_kwargs = mask_kwargs or {}

    A = to_float01(imread(A_path))
    B = to_float01(imread(B_path))

    # --- optional preprocessing ---
    if preprocess == "mountain_center":
        A, B = crop_match_mountain_center(
            A, B,
            place_rel_y=mask_kwargs.get("place_rel_y", 0.30),
            search_top_rel=mask_kwargs.get("search_top_rel", 0.60)
        )
    else:
        A, B = center_crop_to_match(A, B)

    H, W = A.shape[:2]

    # --- choose mask ---
    if mask_kind == "vertical":
        M = soft_vertical_mask(H, W, **{
            "center": mask_kwargs.get("center", 0.5),
            "width":  mask_kwargs.get("width", 0.12),
            "left_is_A": mask_kwargs.get("left_is_A", True)
        })
    elif mask_kind == "triangle":
        M = triangle_mask(
            H, W,
            apex_rel_y   = mask_kwargs.get("apex_rel_y", 0.22),
            base_rel_y   = mask_kwargs.get("base_rel_y", 0.86),
            base_half_rel= mask_kwargs.get("base_half_rel", 0.38),
            feather_px   = mask_kwargs.get("feather_px", 60),
            A_inside     = mask_kwargs.get("A_inside", True)
        )
    else:
        raise ValueError("unknown mask_kind")


    # stacks (for visualization only, not strictly required)
    GA, LA = laplacian_stack(A, levels, sigma)
    GB, LB = laplacian_stack(B, levels, sigma)
    GM = mask_stack(M, levels, sigma)

    # blend
    blended, (LA2, LB2, GM2), blended_levels = multires_blend(A, B, M, levels, sigma)

    # save essentials
    save_img(f"{out_dir}/A.jpg", A)
    save_img(f"{out_dir}/B.jpg", B)
    save_img(f"{out_dir}/mask.jpg", vis01(M))
    save_img(f"{out_dir}/blended.jpg", blended)

    # save stacks (optional but useful for the report)
    for i in range(levels):
        save_img(f"{out_dir}/LA_{i}.jpg", vis01(LA[i]))
        save_img(f"{out_dir}/LB_{i}.jpg", vis01(LB[i]))
        save_img(f"{out_dir}/GM_{i}.jpg", vis01(GM[i]))

    print(f"[custom] saved to: {out_dir}")

# -------------------- main --------------------
if __name__ == "__main__":
    IMG = "projects/project2/images"

    # --- Oraple (recreates Fig. 3.42 a–l) ---
    run_oraple(
        apple_path=f"{IMG}/apple.jpeg",
        orange_path=f"{IMG}/orange.jpeg",
        out_dir="projects/project2/outputs/part2_stacks/oraple",
        levels=5,         # need ≥5 to reference levels 0,2,4
        sigma=2.0,        # per-level blur amount (1.6–2.5 reasonable)
        mask_center=0.50,
        mask_width=0.12
    )

    # --- Custom blend 1: Spring ↔ Winter (vertical seam, soft) ---
    run_custom_blend(
        A_path=f"{IMG}/spring.jpg",
        B_path=f"{IMG}/winter.jpg",
        out_dir="projects/project2/outputs/part2_stacks/custom_spring_winter",
        levels=6,
        sigma=2.0,
        mask_kind="vertical",
        mask_kwargs=dict(center=0.52, width=0.14, left_is_A=True)
    )

    # --- Custom blend 2: Mountain ↔ Skyline (triangle mask) ---
    run_custom_blend(
        A_path=f"{IMG}/mountain.jpg",
        B_path=f"{IMG}/skyline.jpeg",
        out_dir="projects/project2/outputs/part2_stacks/custom_mountain_skyline",
        levels=6,
        sigma=2.0,
        preprocess="mountain_center",   # <- align crop around the cone
        mask_kind="triangle",           # <- centered triangle
        mask_kwargs=dict(
            # where the peak should land in the final crop (0=top, 1=bottom)
            place_rel_y=0.30,           # move up/down if the apex still feels too high/low
            # triangle shape (tweak if needed)
            apex_rel_y=0.22,
            base_rel_y=0.86,
            base_half_rel=0.38,
            feather_px=70,
            A_inside=True               # “inside triangle = mountain”
        )
    )

