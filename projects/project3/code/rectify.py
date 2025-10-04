# projects/project3/code/rectify.py
import numpy as np
from PIL import Image
from .compute_homography import computeH
from .warp import warpImageBilinear

def rectify_single(image_path, src_pts_xy, dst_w, dst_h, out_path):
    """
    src_pts_xy: 4x2 im-Koordinaten (z.B. die Ecken eines Posters)
    Zielrechteck: (0,0)-(w-1,h-1)
    """
    I = Image.open(image_path).convert('RGB')
    dst = np.array([[0,0],[dst_w-1,0],[0,dst_h-1],[dst_w-1,dst_h-1]], dtype=float)
    H = computeH(np.array(src_pts_xy, float), dst)
    warped, _ = warpImageBilinear(I, H, out_bounds=(0,0,dst_w-1,dst_h-1))
    Image.fromarray(np.clip(warped,0,255).astype(np.uint8)).save(out_path)
    return H
