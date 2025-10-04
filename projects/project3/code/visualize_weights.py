#!/usr/bin/env python3
# Visualisiert 3 Gewichte (links/zentral/rechts) mit weichem Cosine-Falloff.
# Speichert Graustufen-PNGs unter outputs/a4_mosaics/weights/.

from pathlib import Path
import numpy as np
from PIL import Image

def _hann2d(h, w):
    # sanfter Top/Bottom-Falloff, damit keine horizontalen Nähte sichtbar sind
    y = np.hanning(max(h, 3)).astype(np.float32)
    x = np.hanning(max(w, 3)).astype(np.float32)
    wy = (y / y.max())[:, None]
    wx = (x / x.max())[None, :]
    return wy * wx  # (h,w)

def _dir_windows(w):
    """1D Fenster entlang x: left-heavy w1x, center w2x, right-heavy w3x"""
    x = np.linspace(0, 1, w, dtype=np.float32)
    # Raised-cosine (Hann-artig): 1→0 von links nach rechts
    w1x = 0.5 * (1.0 + np.cos(np.pi * x))       # links hoch, rechts 0
    w3x = w1x[::-1].copy()                       # gespiegelt: rechts hoch
    # Zentrum: Peak in der Mitte, 0 an den Rändern
    w2x = np.sin(np.pi * x)                      # 0..1..0
    return w1x, w2x, w3x

def make_three_weights(h=360, w=640, center_boost=1.0):
    """
    Erzeugt 3 Gewichtsbilder w1 (links), w2 (zentral), w3 (rechts), jeweils (h,w).
    center_boost >1.0 bevorzugt das Center-Bild leicht.
    """
    base = _hann2d(h, w)                         # Top/Bottom-Feather
    w1x, w2x, w3x = _dir_windows(w)              # Links/Mitte/Rechts entlang x

    w1 = base * w1x[None, :]
    w2 = base * (w2x[None, :] ** 1.0) * center_boost
    w3 = base * w3x[None, :]

    den = w1 + w2 + w3 + 1e-8                    # pro Pixel normieren
    return w1/den, w2/den, w3/den                # alle in [0,1], Summe ≈ 1

def save_gray(img01, path):
    arr = (np.clip(img01, 0, 1) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)

def main():
    root = Path(__file__).resolve().parents[1]            # .../projects/project3
    outd = root / "outputs" / "a4_mosaics" / "weights"
    outd.mkdir(parents=True, exist_ok=True)

    # Größe darfst du anpassen; 360x640 passt gut zur Demo
    w1, w2, w3 = make_three_weights(h=360, w=640, center_boost=1.15)

    save_gray(w1, outd / "weight_w1_left.png")
    save_gray(w2, outd / "weight_w2_center.png")
    save_gray(w3, outd / "weight_w3_right.png")

    print("[saved]", outd)

if __name__ == "__main__":
    main()
