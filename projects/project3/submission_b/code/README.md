# Project 3 — **Part B: Feature Matching & Autostitching**

*(CS180 • Image Warping & Mosaicing — Part B only)*

This README explains the **project structure**, what each script does, **how the pieces from Part A are reused**, and exactly **how to run** everything end-to-end to produce your **B.3 matches** and **B.4 RANSAC mosaics**.

---

## 0) Repository layout (relevant to Part B)

```
projects/
└─ project3/
   ├─ images/                     # input photos (left/center/right per scene)
   │   ihouse_left.jpg
   │   ihouse_center.jpg
   │   ihouse_right.jpg
   │   stadium_low_left.jpg
   │   ... (add your own scenes the same way)
   │
   ├─ code/
   │  ├─ b1_harris_anms.py        # Harris & ANMS primitives (also mirrored in harris_anms.py)
   │  ├─ descriptors.py           # 40×40 → 8×8 descriptor utilities
   │  ├─ b3_match.py              # Part B.3 — full pipeline + saves visualizations + NPZ matches
   │  ├─ b4_ransac_from_b3.py     # Part B.4 — 4pt-RANSAC, homographies, mosaics (uses B.3 NPZ)
   │  ├─ b4_ransac_mosaic.py      # (alt runner) RANSAC + mosaic; similar to the above
   │  ├─ compute_homography.py    # DLT utilities (used elsewhere)
   │  ├─ harris_anms.py           # Harris + ANMS helpers (overlap with b1_harris_anms.py)
   │  ├─ main_b1_harris_anms.py   # convenience CLI for B.1 visualization
   │  ├─ main_b2_descriptors.py   # convenience CLI for B.2 visualization
   │  ├─ main_mosaic_laplacian.py # Part A: Laplacian blend (reused by B.4 when requested)
   │  ├─ mosaic_weighted.py       # Part A: distance-transform feathering helpers
   │  ├─ warp.py, rectify.py ...  # Part A warping utilities (canvas, inverse map, masks)
   │  └─ (other Part A helpers used by the HTML report)
   │
   └─ outputs/
      ├─ b1_features/             # B.1: Harris heatmaps/peaks/ANMS visualizations
      ├─ b2_descriptors/          # B.2: 40×40 and 8×8 patch panels
      ├─ b3_matches/              # B.3: match images + *.npz with exact correspondences
      └─ b4_autostitch/           # B.4: “all matches / inliers / mosaic / manual-vs-auto”
```

**Naming convention for images** (per scene `SCENE_KEY`):

* `SCENE_KEY_left.jpg`, `SCENE_KEY_center.jpg`, `SCENE_KEY_right.jpg` (e.g., `ihouse_left.jpg`).
* You can add new scenes by dropping three images with that pattern into `images/` and passing the keys on the CLI.

---

## 1) Environment / Requirements

Use Python 3.9+ (3.10/3.11 also fine). Install:

```bash
pip install numpy opencv-python pillow scipy
```

* **NumPy**: arrays/linear algebra
* **OpenCV**: gradients, homographies, warping, drawing
* **Pillow**: reading/writing RGB, side-by-side comparisons
* **SciPy**: distance transform & Gaussian blurs (for Part A blending reused in B.4)

> **Tip (Apple Silicon)**: if you hit issues with `opencv-python`, install `opencv-python-headless`.

---

## 2) What’s implemented (and reused from Part A)

### B.1 — Harris + ANMS

* Harris response (R = \det(\mathbf M) - k,\operatorname{tr}(\mathbf M)^2) with Sobel gradients.
* Non-maximum suppression + **ANMS** to keep **strong** and **well-spread** corners.

**Where:** `b1_harris_anms.py` / `harris_anms.py` (used by later steps and for figures).

**Output preview runner:**

```bash
python projects/project3/code/main_b1_harris_anms.py \
  --repo_root . \
  --scenes ihouse stadium_low stadium_high \
  --anms_N 800
```

Creates `outputs/b1_features/*` heatmaps/overlays.

---

### B.2 — Descriptors (40×40 → 8×8)

* For each ANMS point: sample a **40×40** patch (subpixel, Gaussian blur), **downsample to 8×8**, then **bias/gain normalize** (zero mean, unit variance).
* Descriptor is a 64-D vector per keypoint.

**Where:** `descriptors.py` (and embedded copies in `b3_match.py`).

**Output preview runner:**

```bash
python projects/project3/code/main_b2_descriptors.py \
  --repo_root . \
  --scenes ihouse stadium_low stadium_high
```

Creates `outputs/b2_descriptors/*` patch panels.

---

### B.3 — Matching (SSD + Lowe ratio) and **saving NPZ**

* Compute pairwise **SSD** in 64-D descriptor space.
* **Lowe ratio** (1NN/2NN) to filter ambiguous matches (default ~0.5).
* Save **visualizations** of all matches and **top-K** (e.g., 15).
* **Save the exact correspondences** you matched as `*.npz` so **B.4** can reuse them 1:1 (no drift).

**Run:**

```bash
python projects/project3/code/b3_match.py \
  --repo_root . \
  --scenes ihouse stadium_low stadium_high \
  --ratio 0.5 \
  --anms_N 800
```

**Outputs go to** `outputs/b3_matches/`, e.g.

```
ihouse_L_to_C_matches_top15.png
ihouse_R_to_C_matches_top15.png
ihouse_LC_matches_data.npz   # ← used by B.4
ihouse_RC_matches_data.npz   # ← used by B.4
```

> **Coordinate convention:** The `.npz` stores `src_xy` and `dst_xy` in **(x,y)** image coordinates (what the homography code expects). Visualization functions may work in (y,x); the conversion is handled in the scripts when saving.

---

### B.4 — 4-point **RANSAC** Homographies + **Auto Mosaic** (reusing B.3)

* Load the saved **B.3 matches** (`*_matches_data.npz`).
  If a file is missing, we **fall back** to running the B.3 pipeline internally.
* Run **RANSAC**:

  * Sample 4 pairs, compute (H) via normalized DLT, count inliers by reprojection error.
  * Keep the (H) with the largest consensus, then **refit** using all inliers.
* Build a **canvas** large enough for all warped images, **inverse-warp** into it (bilinear), keep **valid masks**.
* **Blend**:

  * By default the script uses **distance-transform feathering** (self-contained).
  * If you pass `--blend laplacian` *and* Part-A Laplacian utilities are available, it reuses them automatically. Otherwise it prints a warning and falls back to feathering.

**Run (typical):**

```bash
python projects/project3/code/b4_ransac_from_b3.py \
  --repo_root . \
  --scenes ihouse stadium_low stadium_high \
  --iters 3000 \
  --thresh 3.0 \
  --topk 15 \
  --blend laplacian
```

* `--iters` : RANSAC iterations
* `--thresh`: inlier threshold in pixels (e.g., 3.0)
* `--topk`  : cap to best-K matches (sorted by Lowe ratio) **per direction** before RANSAC
* `--blend` : `laplacian` (reuse Part-A) or `feather` (built-in fallback)

**Outputs go to** `outputs/b4_autostitch/`, e.g.

```
ihouse_LC_matches_all.jpg
ihouse_LC_matches_inliers.jpg
ihouse_RC_matches_all.jpg
ihouse_RC_matches_inliers.jpg
ihouse_auto_mosaic.jpg
ihouse_manual_vs_auto.jpg    # if a Part-A manual mosaic is found
```

> **“Part-A blending not found; using feather fallback.”**
> This isn’t an error. It just means the script didn’t detect the Part-A Laplacian helper. To use it, keep `main_mosaic_laplacian.py` and its helpers in `code/` and pass `--blend laplacian` as shown.

---

## 3) How the pieces connect

* **B.1/B.2 → B.3**: Harris+ANMS decide *where* we look; descriptors decide *what* we compare.
* **B.3 → B.4**: Matching **saves** `src_xy`/`dst_xy` in `outputs/b3_matches/*.npz`. **B.4 loads these exact pairs**, optionally trims to `--topk`, and runs RANSAC.
* **Part A → B.4**: Once we have (H_{L\to C}) and (H_{R\to C}), B.4 reuses the **warping & blending ideas** from Part A:

  * **Canvas computation** (transform corners, pad with a translation so coordinates are non-negative).
  * **Inverse mapping** with **bilinear** sampling and validity masks.
  * **Blending**:

    * **Feather** via distance-to-boundary weights (DT) — *built-in in B.4*.
    * **Laplacian pyramid** — *reused from Part A* (multi-scale masks to hide both sharp seams and slow luminance drift).

---

## 4) Math (what the code actually computes)

**Harris:** (R = \det(\mathbf M) - k,\mathrm{tr}(\mathbf M)^2) with (\mathbf M = G_\sigma * \begin{bmatrix}I_x^2 & I_x I_y\ I_x I_y & I_y^2\end{bmatrix}).
**ANMS:** keep points with large suppression radius to enforce spatial spread.

**Descriptor:** for each keypoint, sample a **40×40** patch (subpixel), blur, **downsample to 8×8**, then **normalize**: (\tilde{\mathbf d} = (\mathbf d - \mu)/(\sigma+\varepsilon)).

**Matching:** SSD in 64-D, **Lowe ratio** (\rho = d_{1\mathrm{NN}}/d_{2\mathrm{NN}} < \tau) (e.g., (\tau=0.5)).

**Homography (normalized DLT):**
Normalize points by similarities (T_1, T_2) (zero mean, avg. dist (\sqrt{2})); build (A,\mathbf h=0) with rows
[
[0,0,0,-x,-y,-1, y' x, y' y, y'],,\quad [x,y,1,0,0,0,-x' x,-x' y,-x']
]
Solve by SVD (last right singular vector), then denormalize (H=T_2^{-1}\hat H T_1).

**RANSAC:** repeat (N) times: sample 4 pairs, compute (H), mark inliers if
(|,\pi(H[x_i,y_i,1]^\top) - [x'_i,y'_i]^\top,|_2 < \tau) with (\pi([u,v,w])=[u/w,v/w]). Keep the model with the largest consensus; **refit** (H) on all inliers.

**Blending:** build per-image weights from **distance to boundary** (DT), blur/shape, **normalize per pixel**; or blend **Laplacian bands** with Gaussian-blurred masks.

---

## 5) Step-by-step: clean run

From the repo root (where `projects/project3/` lives):

1. **(Optional) B.1 & B.2 figures**

   ```bash
   python projects/project3/code/main_b1_harris_anms.py --repo_root . --scenes ihouse stadium_low stadium_high --anms_N 800
   python projects/project3/code/main_b2_descriptors.py --repo_root . --scenes ihouse stadium_low stadium_high
   ```

2. **B.3: matches + NPZ correspondences**

   ```bash
   python projects/project3/code/b3_match.py \
     --repo_root . \
     --scenes ihouse stadium_low stadium_high \
     --ratio 0.5 \
     --anms_N 800
   ```

   Check `outputs/b3_matches/` for `*_matches_top15.png` and `*_matches_data.npz`.

3. **B.4: RANSAC homographies + auto mosaics**

   ```bash
   python projects/project3/code/b4_ransac_from_b3.py \
     --repo_root . \
     --scenes ihouse stadium_low stadium_high \
     --iters 3000 \
     --thresh 3.0 \
     --topk 15 \
     --blend laplacian
   ```

   Results land in `outputs/b4_autostitch/` (all/inliers/mosaic/manual-vs-auto).

> You can change `--scenes` to just `ihouse` during debugging.

---

## 6) Where each function is used

* **Harris / ANMS**:
  Implemented in `b1_harris_anms.py` and `harris_anms.py`. Called directly in `b3_match.py` (and mirrored inside for portability).
* **Descriptors (40→8 with normalization)**:
  `descriptors.py` (also embedded in `b3_match.py`), used in B.3 to build the 64-D vectors.
* **Matching (SSD + Lowe ratio)**:
  Implemented in `b3_match.py`. Produces match visualizations **and** `*_matches_data.npz`.
* **RANSAC + DLT**:
  Implemented in `b4_ransac_from_b3.py` (local DLT and RANSAC functions). Loads **B.3 NPZ** by default.
* **Warping, masks, canvas**:
  In B.4 they are implemented locally (and are equivalent to Part-A logic).
  If `--blend laplacian` is set, B.4 **reuses Part-A** Laplacian blending from `main_mosaic_laplacian.py`/`mosaic_weighted.py`; otherwise it uses its built-in distance-transform feather.
* **HTML report**: your report page reads images directly from `outputs/*` and doesn’t influence computation.

---

## 7) Troubleshooting

* **“Part-A blending not found; using feather fallback.”**
  Informational. If you want multi-scale Laplacian blending in B.4, keep `main_mosaic_laplacian.py` (and SciPy installed) and run with `--blend laplacian`.

* **OpenCV `imwrite` error (missing required argument `img`)**
  This usually means a stray trailing comma created a tuple, e.g.
  `cv2.imwrite(str(path), image),`  ← remove the comma.

* **Different color look in match images**
  B.3/B.4 draw on BGR images for line overlays and then save. Final mosaics are saved as RGB→BGR for OpenCV. All pipelines save 8-bit sRGB JPEG/PNG.

* **RANSAC fails or produces a wild mosaic**

  * Try stricter matches: lower `--ratio` in B.3 (e.g., 0.45) and/or smaller `--topk` in B.4 (e.g., 10).
  * Increase `--iters` (e.g., 5000) or relax `--thresh` to 4.0 if the scene has mild parallax.

* **No `*_matches_data.npz` found**
  Run B.3 first. B.4 will fall back to recomputing matches, but saving them helps keep Part B deterministic.

---

## 8) Extending to a new scene

1. Drop `myplace_left.jpg`, `myplace_center.jpg`, `myplace_right.jpg` into `projects/project3/images/`.
2. Run:

   ```bash
   python projects/project3/code/b3_match.py --repo_root . --scenes myplace
   python projects/project3/code/b4_ransac_from_b3.py --repo_root . --scenes myplace --iters 3000 --thresh 3.0 --topk 15 --blend laplacian
   ```
3. Check `outputs/b3_matches/` and `outputs/b4_autostitch/`.

---

## 9) Reproducibility notes

* Randomness appears only in **RANSAC sampling**; you can pass `--seed` to `b4_ransac_from_b3.py` to fix it.
* All saved `*.npz` files store the exact correspondences used for the figures/mosaics.

---

## 10) One-liner cheatsheet

```bash
# Generate matches (and the NPZ that B.4 will reuse)
python projects/project3/code/b3_match.py --repo_root . --scenes ihouse stadium_low stadium_high

# Build RANSAC mosaics from those matches (cap to 15 best per direction)
python projects/project3/code/b4_ransac_from_b3.py \
  --repo_root . --scenes ihouse stadium_low stadium_high \
  --iters 3000 --thresh 3.0 --topk 15 --blend laplacian
```

That’s it—Part B will produce the **match visualizations**, **RANSAC inlier plots**, and the **automatic mosaics** that you can drop straight into your write-up page.
