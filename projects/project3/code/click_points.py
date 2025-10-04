import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image

def _load_rgb(path):
    return np.array(Image.open(path).convert('RGB'))

def _ginput_fixed(ax, n, title):
    ax.set_title(title)
    pts = plt.ginput(n, timeout=0)  # n Klicks, beliebig viel Zeit
    ax.set_title('')
    return np.array(pts, dtype=float)

def click_correspondences(imgA_path, imgB_path, n_points=10, out_json=None, show_numbers=True):
    """
    Klicke n_points Korrespondenzen: zuerst ALLE Punkte in Bild A, dann in Bild B,
    in identischer Reihenfolge. Speichert als JSON {"A":[[x,y],...], "B":[[x',y'],...]}.
    """
    A = _load_rgb(imgA_path); B = _load_rgb(imgB_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6), constrained_layout=True)
    ax1.imshow(A); ax1.axis('off'); ax1.set_title('Click in A (left) — {} points'.format(n_points))
    ax2.imshow(B); ax2.axis('off'); ax2.set_title('Click in B (right) — {} points'.format(n_points))

    # Klicks sammeln
    print("[A] Click {} points in LEFT panel (A), ENTER when done.".format(n_points))
    ptsA = _ginput_fixed(ax1, n_points, f'Click {n_points} pts in A (left), press ENTER')
    ax1.scatter(ptsA[:,0], ptsA[:,1], s=30, c='lime')
    for i,(x,y) in enumerate(ptsA):
        if show_numbers: ax1.text(x, y, str(i+1), color='yellow', fontsize=10)

    print("[B] Click the SAME {} points in RIGHT panel (B), same order, ENTER when done.".format(n_points))
    ptsB = _ginput_fixed(ax2, n_points, f'Click {n_points} matching pts in B (right), press ENTER')
    ax2.scatter(ptsB[:,0], ptsB[:,1], s=30, c='cyan')
    for i,(x,y) in enumerate(ptsB):
        if show_numbers: ax2.text(x, y, str(i+1), color='yellow', fontsize=10)

    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump({"A": ptsA.tolist(), "B": ptsB.tolist()}, f, indent=2)
        print(f"[saved] {out_json}")

    plt.show()
    return ptsA, ptsB
