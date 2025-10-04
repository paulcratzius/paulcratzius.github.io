# projects/project3/code/run_homography_pair.py
import os, json
from pathlib import Path
import numpy as np

# Robuste Imports: zuerst lokal (direkt gestartet), optional Paket-Import als Fallback
try:
    from compute_homography import computeH, reprojection_rmse
    from vis_correspondences import visualize_pairs
except Exception:
    # Fallback, falls das File als Skript direkt gestartet wird
    from projects.project3.code.compute_homography import computeH, reprojection_rmse
    from projects.project3.code.vis_correspondences import visualize_pairs


def run_pair(imgA: str, imgB: str, pairs_json: str, out_dir: str, tag: str) -> dict:
    """
    Rechnet H (imgA -> imgB) aus Korrespondenzen, speichert H, RMSE und Korrespondenz-Overlay.

    Args:
        imgA: Pfad zu Quellbild (left/right)
        imgB: Pfad zu Zielbild (center)
        pairs_json: JSON mit Schlüsseln {"A":[[x,y],...], "B":[[x',y'],...]}
        out_dir: Ausgabeverzeichnis
        tag: kurzer Name fürs Benennen der Dateien (z.B. ihouse_L_to_C)

    Returns:
        dict mit Pfaden und Metriken
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with open(pairs_json, "r") as f:
        d = json.load(f)
    A = np.array(d["A"], dtype=float)
    B = np.array(d["B"], dtype=float)

    # 1) H berechnen (DLT + Normalisierung)
    H = computeH(A, B)

    # 2) Reprojektion-Fehler
    rmse, errs = reprojection_rmse(H, A, B)

    # 3) Speichern
    H_path   = out_dir / f"H_{tag}.txt"
    stats_js = out_dir / f"stats_{tag}.json"
    pairs_png= out_dir / f"{tag}_pairs.png"

    np.savetxt(H_path, H, fmt="%.6f")
    with open(stats_js, "w") as f:
        json.dump({"rmse": float(rmse),
                   "per_point_err": [float(x) for x in errs]}, f, indent=2)

    # 4) Visualisierung der Korrespondenzen
    visualize_pairs(imgA, imgB, pairs_json, str(pairs_png))

    print(f"[H]   saved: {H_path}")
    print(f"[RMSE] {rmse:.3f}px  (max per-point {errs.max():.3f}px)")
    print(f"[viz] saved: {pairs_png}")

    return {"H_path": str(H_path),
            "stats_path": str(stats_js),
            "viz_path": str(pairs_png),
            "rmse": float(rmse),
            "max_err": float(errs.max())}
