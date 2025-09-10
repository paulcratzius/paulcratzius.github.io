# demo-emir-ssd.py
import os, cv2, numpy as np
from solution import process_one

if __name__ == "__main__":
    # Melons mit single-ssd (L2 Norm)
    res = process_one("projects/project1/images/large/melons.tif", method="single-ssd")

    out_dir = "projects/project1/outputs/demos"
    os.makedirs(out_dir, exist_ok=True)   # <-- ersetzt ensure_dir

    out_path = os.path.join(out_dir, "melons_single-ssd.jpg")
    rgb = (np.clip(res['rgb'], 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("wrote:", out_path)
