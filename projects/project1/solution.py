
import numpy as np, cv2, os, time
from metrics import ssd, ncc, phase_correlation, crop_inner, overlap_views
from pyramid import align_pyramid_edges, crop_borders



def load_image(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)

    # Falls 3-kanalig (manche JPGs), nach Graustufe konvertieren
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # -> 2D

    # float32 und auf [0,1] skalieren (funktioniert für uint8/uint16)
    im = im.astype(np.float32)
    maxv = im.max()
    if maxv > 0:
        im /= maxv

    H = im.shape[0]
    h = int(np.floor(H / 3.0))

    # Vertikales Splitten (jetzt 2D-Indexierung!)
    b = im[:h, :]
    g = im[h:2*h, :]
    r = im[2*h:3*h, :]

    return b, g, r

def shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    # zyklisches Verschieben ohne Größenänderung
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)

def compose_rgb(b: np.ndarray, g: np.ndarray, r: np.ndarray,
                dy_g: int, dx_g: int, dy_r: int, dx_r: int) -> np.ndarray:
    # G und R an B ausrichten und zu RGB stapeln
    g_al = shift_image(g, dy_g, dx_g)
    r_al = shift_image(r, dy_r, dx_r)
    rgb = np.dstack([r_al, g_al, b])  # Reihenfolge: R, G, B
    return rgb.astype(np.float32)



def align_single(ref, tgt, win=15, metric='ssd', crop_frac=0.10):
    best = None
    ref_c = crop_inner(ref, crop_frac)
    tgt_c = crop_inner(tgt, crop_frac)
    Hc, Wc = ref_c.shape

    for dy in range(-win, win+1):
        for dx in range(-win, win+1):
            # auf dem gecroppten Bereich vergleichen → robust gg. Ränder
            ov_r, ov_t = overlap_views(ref_c, tgt_c, dy, dx)
            if ov_r is None: 
                continue
            A = ref_c[ov_r]
            B = tgt_c[ov_t]
            if metric == 'ssd':
                score = -ssd(A, B)  # höher ist besser
            elif metric == 'ncc':
                score = ncc(A, B)
            else:
                raise ValueError('unknown metric')
            if best is None or score > best[0]:
                best = (score, dy, dx)
    if best is None:
        return 0, 0, 0.0
    _, dy, dx = best
    return dy, dx, best[0]



def process_one(path, method='single-ssd', base_win=20, crop_fine=0.12, crop_coarse=0.22):
    b, g, r = load_image(path)
    # 1) Ränder entfernen: je Seite 10% (insgesamt 20% in Höhe/Breite)
    border_frac = 0.20
    b = crop_borders(b, border_frac)
    g = crop_borders(g, border_frac)
    r = crop_borders(r, border_frac)
    t0 = time.time()

    if method == 'single-ssd':
        dyg, dxg, _ = align_single(b, g, win=15, metric='ssd', crop_frac=0.10)
        dyr, dxr, _ = align_single(b, r, win=15, metric='ssd', crop_frac=0.10)

    elif method == 'single-ncc':
        dyg, dxg, _ = align_single(b, g, win=15, metric='ncc', crop_frac=0.10)
        dyr, dxr, _ = align_single(b, r, win=15, metric='ncc', crop_frac=0.10)

    elif method == 'phase':
        dyg, dxg = phase_correlation(b, g)
        dyr, dxr = phase_correlation(b, r)

    elif method == 'pyramid-ssd':
        dyg, dxg = align_pyramid_edges(b, g, metric='ssd', base_win=20,
                                    border_frac=0.12, target_min=120,
                                    edge_method='central')   # oder 'prewitt'
        dyr, dxr = align_pyramid_edges(b, r, metric='ssd', base_win=20,
                                    border_frac=0.12, target_min=120,
                                    edge_method='central')

    elif method == 'pyramid-ncc':
        dyg, dxg = align_pyramid_edges(b, g, metric='ncc', base_win=20,
                                    border_frac=0.12, target_min=120,
                                    edge_method='central')
        dyr, dxr = align_pyramid_edges(b, r, metric='ncc', base_win=20,
                                    border_frac=0.12, target_min=120,
                                    edge_method='central')

    else:
        raise ValueError('unknown method')

    rgb = compose_rgb(b, g, r, dyg, dxg, dyr, dxr)
    dt = time.time() - t0
    return {'path': path, 'method': method,
            'dy_g': dyg, 'dx_g': dxg, 'dy_r': dyr, 'dx_r': dxr,
            'time_sec': dt, 'rgb': rgb}


