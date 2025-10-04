# projects/project3/code/vis_correspondences.py
import json, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def _load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def visualize_pairs(imgA_path, imgB_path, json_pairs_path, out_png):
    with open(json_pairs_path, 'r') as f:
        data = json.load(f)
    Apts = np.array(data['A'], dtype=float)
    Bpts = np.array(data['B'], dtype=float)
    A = _load_rgb(imgA_path); B = _load_rgb(imgB_path)

    h = max(A.shape[0], B.shape[0])
    wA, wB = A.shape[1], B.shape[1]
    canvas = np.ones((h, wA+wB, 3), dtype=np.uint8)*255
    canvas[:A.shape[0], :wA] = A
    canvas[:B.shape[0], wA:wA+wB] = B

    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(canvas); ax.axis('off')

    rng = np.random.default_rng(0)
    colors = rng.random((len(Apts), 3))
    for i, ((x,y),(xp,yp)) in enumerate(zip(Apts, Bpts), start=1):
        c = colors[i-1]
        ax.plot([x, xp+wA], [y, yp], '-', color=c, linewidth=1.5)
        ax.scatter([x, xp+wA], [y, yp], s=18, color=c)
        ax.text(x+3, y, str(i), color=c, fontsize=9)
        ax.text(xp+wA+3, yp, str(i), color=c, fontsize=9)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_png}")
