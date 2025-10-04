#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np

from .compute_homography import computeH
from .mosaic_weighted import mosaic_three_weighted

SCENES = ["ihouse", "stadium_low", "stadium_high"]

def main():
    root = Path(__file__).resolve().parents[1]    # .../projects/project3
    imgd = root / "images"
    a2d  = root / "outputs" / "a2"
    outd = root / "outputs" / "a4_weighted"

    for sc in SCENES:
        print(f"[weighted mosaics] {sc}")
        CL = imgd / f"{sc}_center.jpg"
        LL = imgd / f"{sc}_left.jpg"
        RR = imgd / f"{sc}_right.jpg"

        dL = json.load(open(a2d / f"{sc}_left_to_center.json"))
        dR = json.load(open(a2d / f"{sc}_right_to_center.json"))
        HL = computeH(np.array(dL["A"], float), np.array(dL["B"], float))
        HR = computeH(np.array(dR["A"], float), np.array(dR["B"], float))

        # a) Hann/Cosine
        out_hann = outd / "hann"
        weights_dir_hann = out_hann / "weights" / sc
        mosaic_three_weighted(
            CL, LL, RR, HL, HR,
            out_hann / f"{sc}_mosaic_hann.jpg",
            weight_mode="hann",
            center_boost=1.15,
            return_weights_dir=weights_dir_hann
        )

        # b) Distance Transform
        out_dist = outd / "dist"
        weights_dir_dist = out_dist / "weights" / sc
        mosaic_three_weighted(
            CL, LL, RR, HL, HR,
            out_dist / f"{sc}_mosaic_dist.jpg",
            weight_mode="dist",
            center_boost=1.10,
            return_weights_dir=weights_dir_dist
        )
        
        # c) bwdist (DT + blur + gamma)
        out_bwd = outd / "bwdist"
        weights_dir_bwd = out_bwd / "weights" / sc
        mosaic_three_weighted(
            CL, LL, RR, HL, HR,
            out_bwd / f"{sc}_mosaic_bwdist.jpg",
            weight_mode="bwdist",
            center_boost=1.10,                # leichtes Center-Prior
            return_weights_dir=weights_dir_bwd
        )


        print("  ->", out_hann / f"{sc}_mosaic_hann.jpg")
        print("  ->", out_dist / f"{sc}_mosaic_dist.jpg")

if __name__ == "__main__":
    main()
