import cv2
import numpy as np
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent

OBJ_DIR = PROJ_ROOT / "inputs" / "object"
CALIB_PATH = PROJ_ROOT / "outputs" / "calibration" / "camera_params.npz"
POSES_PATH = PROJ_ROOT / "outputs" / "object_poses.npz"

OUT_DIR = PROJ_ROOT / "outputs" / "dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "my_object.npz"

calib = np.load(CALIB_PATH)
K = calib["K"]
dist_coeffs = calib["dist_coeffs"]

pose_data = np.load(POSES_PATH)
c2ws_all = pose_data["c2ws"]          # (N, 4, 4)
filenames_all = pose_data["filenames"]  # (N,)

print("Loaded calibration & poses.")
print("K =\n", K)
print("dist_coeffs =", dist_coeffs.ravel())
print(f"Found poses for {len(filenames_all)} images.")

first_img_path = OBJ_DIR / filenames_all[0]
first_img = cv2.imread(str(first_img_path))
if first_img is None:
    raise RuntimeError(f"Could not read reference image: {first_img_path}")

h_ref, w_ref = first_img.shape[:2]
print(f"Reference image size (h, w): {h_ref}, {w_ref}")

new_K, roi = cv2.getOptimalNewCameraMatrix(
    K, dist_coeffs, (w_ref, h_ref), alpha=0, newImgSize=(w_ref, h_ref)
)
x, y, w_roi, h_roi = roi
print("ROI from undistortion (x, y, w_roi, h_roi):", roi)

new_K[0, 2] -= x
new_K[1, 2] -= y

print("New camera matrix (after cropping) new_K =\n", new_K)

focal = float(new_K[0, 0])  # we take Fx as focal length
print("Using focal =", focal)

#normalize pictures (undistort)
undistorted_images = []
valid_c2ws = []
valid_filenames = []

for fname, c2w in zip(filenames_all, c2ws_all):
    img_path = OBJ_DIR / fname
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[WARN] Could not read {img_path}, skipping.")
        continue

    h, w = img_bgr.shape[:2]

    # 1) size normalization
    if (h, w) != (h_ref, w_ref):
        if (h, w) == (w_ref, h_ref):
            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
            h, w = img_bgr.shape[:2]
            print(f"[INFO] Rotated {fname} to match reference orientation.")
        else:
            img_bgr = cv2.resize(img_bgr, (w_ref, h_ref))
            h, w = img_bgr.shape[:2]
            print(f"[INFO] Resized {fname} to ({h_ref}, {w_ref}).")

    # Sanity-Check
    assert (h, w) == (h_ref, w_ref), f"Unexpected size for {fname}: {(h, w)}"

    # 2) undistort with shared new_K
    undistorted = cv2.undistort(img_bgr, K, dist_coeffs, None, new_K)

    # 3) valid ROI
    undistorted = undistorted[y:y + h_roi, x:x + w_roi]

    # 4) BGR -> RGB
    undistorted_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

    undistorted_images.append(undistorted_rgb)
    valid_c2ws.append(c2w)
    valid_filenames.append(fname)

if len(undistorted_images) == 0:
    raise RuntimeError("No undistorted images produced. Something went wrong.")

images = np.stack(undistorted_images, axis=0)        # (N, H, W, 3)
c2ws = np.stack(valid_c2ws, axis=0)                  # (N, 4, 4)
valid_filenames = np.array(valid_filenames)

N, H_new, W_new, _ = images.shape
print(f"Final undistorted image stack shape: {images.shape}")
print(f"Used {N} images after filtering/normalizing.")

# Train / Val / Test Split
N_train = int(0.8 * N)
N_val = int(0.1 * N)
N_test = N - N_train - N_val

train_indices = np.arange(0, N_train)
val_indices = np.arange(N_train, N_train + N_val)
test_indices = np.arange(N_train + N_val, N)

print(f"Split: N={N}, N_train={N_train}, N_val={N_val}, N_test={N_test}")

images_train = images[train_indices]
c2ws_train = c2ws[train_indices]

if N_val > 0:
    images_val = images[val_indices]
    c2ws_val = c2ws[val_indices]
else:
    images_val = np.empty((0, H_new, W_new, 3), dtype=images.dtype)
    c2ws_val = np.empty((0, 4, 4), dtype=c2ws.dtype)

if N_test > 0:
    c2ws_test = c2ws[test_indices]
else:
    c2ws_test = np.empty((0, 4, 4), dtype=c2ws.dtype)

np.savez(
    OUT_PATH,
    images_train=images_train,    # (N_train, H, W, 3)
    c2ws_train=c2ws_train,        # (N_train, 4, 4)
    images_val=images_val,        # (N_val, H, W, 3)
    c2ws_val=c2ws_val,            # (N_val, 4, 4)
    c2ws_test=c2ws_test,          # (N_test, 4, 4)
    focal=focal,                  # float
)

print(f"\nSaved dataset to {OUT_PATH}")
print("keys in npz:", np.load(OUT_PATH).files)
