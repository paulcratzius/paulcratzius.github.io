#!/usr/bin/env python3
from pathlib import Path
from .make_mosaics import mosaic_from_json

SCENES = ["ihouse", "stadium_low", "stadium_high"]

def main():
    root = Path(__file__).resolve().parents[1]     # .../projects/project3
    a2dir = root/"outputs"/"a2"
    out_m = root/"outputs"/"a4_mosaics"
    out_m.mkdir(parents=True, exist_ok=True)

    for sc in SCENES:
        print(f"[mosaic] {sc}")
        path = mosaic_from_json(a2dir, out_m, sc)
        print("  ->", path)

if __name__ == "__main__":
    main()
