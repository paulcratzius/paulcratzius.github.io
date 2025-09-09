# runner.py
import os, glob, csv, shutil
import numpy as np
import cv2
from typing import Iterable

def save_rgb(out_path: str, rgb: np.ndarray):
    img8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))

def reset_out_dir(out_dir: str):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

def run_batch(in_dir: str, out_dir: str, methods: Iterable[str], process_one_func, csv_log='results.csv'):
    reset_out_dir(out_dir)
    rows = []
    for path in sorted(glob.glob(os.path.join(in_dir, '*'))):
        if os.path.isdir(path):
            continue
        for m in methods:
            print(f"Processing: {path} with {m}")
            res = process_one_func(path, method=m)  # ruft solution.process_one
            print(f" -> {res['time_sec']:.3f} sec")

            name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f'{name}_{m}.jpg')
            save_rgb(out_path, res['rgb'])
            rows.append([name, m, res['dx_g'], res['dy_g'], res['dx_r'], res['dy_r'], res['time_sec'], out_path])

    with open(os.path.join(out_dir, csv_log), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image','method','dx_g','dy_g','dx_r','dy_r','time_sec','out_path'])
        w.writerows(rows)
    return rows
