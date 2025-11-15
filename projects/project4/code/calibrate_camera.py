import cv2
import numpy as np
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent                          # projects/project4
CALIB_DIR = PROJ_ROOT / "inputs" / "calibration"
OUT_DIR = PROJ_ROOT / "outputs" / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

tag_size = 0.055  # meters

objp_single = np.array(
    [
        [0.0,       0.0,       0.0],
        [tag_size,  0.0,       0.0],
        [tag_size,  tag_size,  0.0],
        [0.0,       tag_size,  0.0],
    ],
    dtype=np.float32,
)

objpoints = []  # list of (N_pts, 3)
imgpoints = []  # list of (N_pts, 2)

img_size = None

image_paths = sorted(
    [p for p in CALIB_DIR.iterdir()
     if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
)

print(f"Found {len(image_paths)} calibration images.")

for path in image_paths:
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Could not read {path}, skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        # cv2.calibrateCamera expects (width, height)
        img_size = (gray.shape[1], gray.shape[0])

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print(f"[INFO] No markers in {path.name}, skipping.")
        continue

    c = corners[0].reshape(-1, 2).astype(np.float32)  # (4,2)

    imgpoints.append(c)
    objpoints.append(objp_single.copy())

    debug_img = img.copy()
    cv2.aruco.drawDetectedMarkers(debug_img, [corners[0]], ids[0:1])
    debug_path = OUT_DIR / f"debug_{path.stem}.jpg"
    cv2.imwrite(str(debug_path), debug_img)

print(f"Used {len(imgpoints)} images with detected markers.")

if len(imgpoints) < 5:
    raise RuntimeError("Not enough valid calibration images. Need at least ~5+ with detections.")

print("Calibrating camera ...")
ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None,
)

print("\n=== Calibration Results ===")
print(f"Reprojection error: {ret:.6f}")
print("Camera matrix K:")
print(K)
print("Distortion coefficients:")
print(dist_coeffs.ravel())

out_path = OUT_DIR / "camera_params.npz"
np.savez(
    out_path,
    K=K,
    dist_coeffs=dist_coeffs,
    reprojection_error=ret,
    img_width=img_size[0],
    img_height=img_size[1],
)

print(f"\nSaved camera parameters to {out_path}")
print(f"Debug images with drawn markers are in: {OUT_DIR}")
