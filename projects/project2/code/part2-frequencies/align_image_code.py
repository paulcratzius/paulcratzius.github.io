# align_image_code.py
import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr

# ------------------ Interaktive Punkte ------------------

def get_points(im1, im2):
    """
    Lass die/den Nutzer:in je 2 Punkte pro Bild klicken (z.B. Augen).
    Rückgabe: (p1, p2, p3, p4) mit p1/p2 aus im1 und p3/p4 aus im2.
    """
    print('Please select 2 points in each image for alignment.')
    plt.figure()
    plt.imshow(im1, cmap=None if im1.ndim == 3 else "gray")
    p1, p2 = plt.ginput(2)
    plt.close()

    plt.figure()
    plt.imshow(im2, cmap=None if im2.ndim == 3 else "gray")
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

# ------------------ Hilfsfunktionen ------------------

def _as_float01(arr):
    a = arr.astype(np.float32, copy=False)
    if a.max() > 1.5:
        a /= 255.0
    return np.clip(a, 0.0, 1.0)

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def recenter(im, r, c):
    """
    Pad so, dass (r,c) näher an die Bildmitte verschoben wird.
    'edge' vermeidet schwarze Dreiecke.
    """
    R, C = im.shape[:2]
    rpad = int(abs(2*r + 1 - R))
    cpad = int(abs(2*c + 1 - C))
    pad_h = (0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad)
    pad_w = (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad)

    if im.ndim == 2:
        pads = (pad_h, pad_w)
    else:
        pads = (pad_h, pad_w, (0, 0))
    return np.pad(im, pads, mode='edge')

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)
    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    """
    Skaliert so, dass die Distanz der zwei Punkte in im1 zu der in im2 passt.
    (im1 wird skaliert, wenn dscale<1; sonst im2)
    """
    p1, p2, p3, p4 = pts
    len1 = np.hypot(p2[1] - p1[1], p2[0] - p1[0])
    len2 = np.hypot(p4[1] - p3[1], p4[0] - p3[0])
    dscale = len2 / (len1 + 1e-8)

    if dscale < 1.0:
        im1 = sktr.rescale(
            im1, dscale,
            channel_axis=-1 if im1.ndim == 3 else None,
            anti_aliasing=True, mode='edge', preserve_range=True
        )
    else:
        im2 = sktr.rescale(
            im2, 1.0/dscale,
            channel_axis=-1 if im2.ndim == 3 else None,
            anti_aliasing=True, mode='edge', preserve_range=True
        )
    return _as_float01(im1), _as_float01(im2)

def rotate_im1(im1, im2, pts):
    """
    Rotiere im1 so, dass die Verbindung p1->p2 denselben Winkel hat
    wie p3->p4 in im2. im2 bleibt unverändert.
    """
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta_deg = (theta2 - theta1) * 180.0 / math.pi

    im1r = sktr.rotate(
        im1, dtheta_deg, resize=False, mode='edge', preserve_range=True
    )
    return _as_float01(im1r), dtheta_deg

def match_img_size(im1, im2):
    """
    Bringt beide Bilder per zentriertem Zuschnitt (Crop) auf identische Größe.
    Funktioniert für 2D (H,W) und 3D (H,W,C).
    """
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    H = int(min(h1, h2))
    W = int(min(w1, w2))

    def center_crop(img, H, W):
        h, w = img.shape[:2]
        y0 = max(0, (h - H) // 2)
        x0 = max(0, (w - W) // 2)
        return img[y0:y0+H, x0:x0+W, ...] if img.ndim == 3 else img[y0:y0+H, x0:x0+W]

    im1c = center_crop(im1, H, W)
    im2c = center_crop(im2, H, W)

    # Kanalzahl kompatibel halten (falls eins 2D ist und das andere (H,W,1))
    if im1c.ndim == 2 and im2c.ndim == 3 and im2c.shape[2] == 1:
        im1c = im1c[..., None]
    if im2c.ndim == 2 and im1c.ndim == 3 and im1c.shape[2] == 1:
        im2c = im2c[..., None]

    assert im1c.shape == im2c.shape, f"Shapes differ after crop: {im1c.shape} vs {im2c.shape}"
    return _as_float01(im1c), _as_float01(im2c)

# ------------------ Hauptfunktion ------------------

def align_images(im1, im2):
    """
    Interaktive Ausrichtung via 2 Punktpaaren pro Bild.
    WICHTIG: Diese Funktion verändert IMMER das ERSTE Bild (im1):
      1) Zentrieren durch Padding,
      2) Größenabgleich (Skalierung) mit im2,
      3) Rotation,
      4) gemeinsamer, zentrierter Crop auf gleiche Maße.
    Rückgabe: (im1_aligned, im2_cropped) jeweils float32 in [0,1]
    """
    im1 = _as_float01(im1)
    im2 = _as_float01(im2)

    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, _   = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1.astype(np.float32), im2.astype(np.float32)
