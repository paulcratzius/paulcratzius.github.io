
# Project 2 — Filters, Frequencies & Multiresolution Blending
**Code folder README**

This folder contains the Python code for CS180 Project 2:

- **Part 2.1**: Unsharp masking (image sharpening)  
- **Part 2.2**: Hybrid images (Oliva, Torralba & Schyns, SIGGRAPH 2006)  
- **Part 2.3**: Gaussian & Laplacian **stacks** (no downsampling)  
- **Part 2.4**: Multiresolution blending (Burt & Adelson)

A static web page at `projects/project2/index.html` automatically loads the results from the `outputs/` folders described below.

---

## 1) Environment & Dependencies

Python **3.9+** recommended.

```bash
# (optional) create a virtual env
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install numpy imageio matplotlib scipy opencv-python
```

-----

## 2\) The code expects this layout relative to the repo root:

```
projects/
  project2/
    images/                 # <-- put inputs here (see list below)
      apple.jpeg
      orange.jpeg
      spring.jpg
      winter.jpg
      mountain.jpg
      skyline.jpeg
      DerekPicture.jpg
      nutmeg.jpg
      thomas_hofmann.jpeg
      albert_berger.jpeg
      i-house-pic.JPG
      ruben.jpg
      ... (any others)
    outputs/
      part2_unsharp/        # Part 2.1 results
      part2_hybrid/         # Part 2.2 results (subfolders per pair)
      part2_stacks/
        oraple/             # Part 2.3 Szelski Fig. 3.42 reproduction
        custom_* # Part 2.4 custom blends
    code/
      unsharp_mask.py
      hybrid_images.py
      align_image_code.py
      stacks_and_blend.py
      README.md             # (this file)
    index.html              # project web page (loads from outputs/*)
```

-----

## 3\) How to run

### A) Part 2.1 — Unsharp Masking

**Script**: `projects/project2/code/unsharp_mask.py`  
**Inputs**: images in `projects/project2/images` (e.g., `half-dome.jpg`, `taj.jpg`, etc.)  
**Run**:

```bash
python projects/project2/code/unsharp_mask.py
```

**Outputs**: `projects/project2/outputs/part2_unsharp/`

  - `stem_blur_s{σ}.jpg` (Gaussian blur)
  - `stem_high_s{σ}.jpg` (high-frequency = I - blurσ)
  - `stem_sharp_s{σ}_a{α}.jpg` (sharpened = I + α(I - blurσ))

The web page loader tries a few σ values (e.g., 1.00, 1.60, 2.00) and picks the first file that exists.

-----

### B) Part 2.2 — Hybrid Images

**Scripts**:

  - `projects/project2/code/hybrid_images.py` (main)
  - `projects/project2/code/align_image_code.py` (interactive alignment helper)

**Inputs** (put in `projects/project2/images`):

  - **Required pair**: `DerekPicture.jpg` (low) + `nutmeg.jpg` (high)
  - **Two custom pairs**, e.g.:
      - `thomas_hofmann.jpeg` + `albert_berger.jpeg`
      - `i-house-pic.JPG` + `ruben.jpg`

**Run**:

```bash
python projects/project2/code/hybrid_images.py
```

**Interactive alignment** (per pair):  
The tool asks you to click two corresponding points (e.g., eyes) on the high-pass (B) image. Then click the same two points in the same order on the low-pass (A) image. Close the windows to proceed. We rotate/scale/translate B to A by default (so B is aligned to A).

**Outputs**: `projects/project2/outputs/part2_hybrid/<pair_name>/`

  - `A_original.jpg`, `B_original.jpg`, `A_aligned.jpg`, `B_aligned.jpg`
  - `lowA_sigma{...}.jpg`, `highB_sigma{...}.jpg`
  - `hybrid.jpg` (final)
  - **FFTs**: `fft_A.jpg`, `fft_B.jpg`, `fft_lowA.jpg`, `fft_highB.jpg`, `fft_hybrid.jpg`
  - **Optional visuals**: `gaussian_pyramid.jpg`, `laplacian_pyramid.jpg`
  - `params.txt` (records σ/weights/flags)

**Tips**

  - Hybrid looks too much like B → reduce `w_high`, increase `σ_low`, or use DoG band-pass for B.
  - Hybrid looks too blurred → decrease `σ_low`.
  - Colored halos → keep B’s high-pass in luminance (already implemented).

-----

### C) Part 2.3 — Gaussian & Laplacian Stacks (no downsampling)

**Script**: `projects/project2/code/stacks_and_blend.py`  
**Function**: `run_oraple(...)` (called in `__main__`)  
**Inputs**: `apple.jpeg`, `orange.jpeg` in `projects/project2/images`  
**Run**:

```bash
python projects/project2/code/stacks_and_blend.py
```

**Outputs**: `projects/project2/outputs/part2_stacks/oraple/`

  - **Full stacks per image**: `GA_*.jpg`, `LA_*.jpg`, `GB_*.jpg`, `LB_*.jpg`
  - **Mask stack**: `GM_*.jpg`
  - **Szelski Fig. 3.42-like panels**:
      - (a,b,c) high level; (d,e,f) medium; (g,h,i) low (we pick levels 0/2/4)
      - (j) weighted apple, (k) weighted orange, (l) final `(l)_oraple.jpg`
  - **Inputs & mask**: `apple_input.jpg`, `orange_input.jpg`, `mask.jpg`

You can tune levels (≥5 so 0/2/4 exist) and sigma (\~1.6–2.5).

-----

### D) Part 2.4 — Multiresolution Blending (custom pairs)

**Script**: `projects/project2/code/stacks_and_blend.py`  
**Function**: `run_custom_blend(...)` (twice in `__main__`)  
**Inputs**:

  - `spring.jpg` + `winter.jpg` (vertical soft seam)
  - `mountain.jpg` + `skyline.jpeg` (irregular polygon/triangle mask)

Both images are center-cropped to a common size; for the mountain pair we use a top-anchored crop to keep the ridge and a centered triangular mask (points configurable in `mask_kwargs`).

**Run**:

```bash
python projects/project2/code/stacks_and_blend.py
```

**Outputs**:

  - `projects/project2/outputs/part2_stacks/custom_spring_winter/`
  - `projects/project2/outputs/part2_stacks/custom_mountain_skyline/`

Each folder contains:

  - `A.jpg`, `B.jpg` (inputs after crop)
  - `mask.jpg` (soft mask actually used)
  - `blended.jpg` (final result)
  - **Optional per-level dumps**: `LA_i.jpg`, `LB_i.jpg`, `GM_i.jpg`

<!-- end list -->

```
```