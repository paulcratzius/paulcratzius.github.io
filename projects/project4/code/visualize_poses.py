import time
from pathlib import Path

import cv2
import numpy as np
import viser

THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent

OBJ_DIR = PROJ_ROOT / "inputs" / "object"
POSES_PATH = PROJ_ROOT / "outputs" / "object_poses.npz"
CALIB_PATH = PROJ_ROOT / "outputs" / "calibration" / "camera_params.npz"

pose_data = np.load(POSES_PATH)
c2ws = pose_data["c2ws"]
filenames = pose_data["filenames"]

calib = np.load(CALIB_PATH)
K = calib["K"]

server = viser.ViserServer(share=True)

for i, fname in enumerate(filenames):
    img_path = OBJ_DIR / fname
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    f = K[0, 0]  # fx
    fov = 2 * np.arctan2(H / 2, f)
    aspect = W / H

    c2w = c2ws[i]

    server.scene.add_camera_frustum(
        f"/cameras/{i}",
        fov=float(fov),
        aspect=float(aspect),
        scale=0.02,
        wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3],
        image=img,
    )

print("Open the Viser URL in your browser to inspect the camera frustums.")
while True:
    time.sleep(0.1)
