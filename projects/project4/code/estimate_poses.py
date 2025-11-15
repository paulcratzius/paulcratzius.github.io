import cv2
import numpy as np
from pathlib import Path

# ---------- Pfade ----------
THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent

OBJ_DIR = PROJ_ROOT / "inputs" / "object"
CALIB_PATH = PROJ_ROOT / "outputs" / "calibration" / "camera_params.npz"
OUT_DIR = PROJ_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSES_PATH = OUT_DIR / "object_poses.npz"

# ---------- Kalibrierungsdaten laden ----------
calib = np.load(CALIB_PATH)
K = calib["K"]
dist_coeffs = calib["dist_coeffs"]

print("Loaded calibration:")
print("K =\n", K)
print("dist_coeffs =", dist_coeffs.ravel())

# ---------- ArUco-Setup ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Deine Tag-Größe (wie bei der Kalibrierung): 5.5 cm
tag_size = 0.055  # Meter

# 3D-Koordinaten der vier Tag-Ecken im Weltkoordinatensystem (z=0)
objp_single = np.array(
    [
        [0.0,       0.0,       0.0],
        [tag_size,  0.0,       0.0],
        [tag_size,  tag_size,  0.0],
        [0.0,       tag_size,  0.0],
    ],
    dtype=np.float32,
)

image_paths = sorted(
    [p for p in OBJ_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
)

print(f"Found {len(image_paths)} object images.")

c2ws = []
used_filenames = []

for path in image_paths:
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Could not read {path}, skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        print(f"[INFO] No markers detected in {path.name}, skipping.")
        continue

    # Wir nehmen an, es gibt genau EIN Tag im Bild → corners[0]
    corners_2d = corners[0].reshape(-1, 2).astype(np.float32)  # (4, 2)

    # Pose schätzen (world-to-camera)
    success, rvec, tvec = cv2.solvePnP(
        objp_single,
        corners_2d,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        print(f"[WARN] solvePnP failed for {path.name}, skipping.")
        continue

    R, _ = cv2.Rodrigues(rvec)  # 3x3
    t = tvec.reshape(3, 1)

    # world-to-camera Matrix (4x4)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = t[:, 0]

    # camera-to-world (invertieren)
    c2w = np.linalg.inv(w2c).astype(np.float32)

    c2ws.append(c2w)
    used_filenames.append(path.name)

    # Optional: einmal pro Bild ein kleines Log
    print(f"[OK] Pose for {path.name}")

if len(c2ws) == 0:
    raise RuntimeError("No valid poses estimated. Check that the tag is visible and detectable.")

c2ws = np.stack(c2ws, axis=0)  # (N, 4, 4)
used_filenames = np.array(used_filenames)

np.savez(
    POSES_PATH,
    c2ws=c2ws,
    filenames=used_filenames,
)

print("\n=== Pose Estimation Done ===")
print(f"Saved poses for {c2ws.shape[0]} images to {POSES_PATH}")
