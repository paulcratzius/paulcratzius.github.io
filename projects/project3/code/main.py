#!/usr/bin/env python3
# projects/project3/code/main_a2_all.py
# Klickt Korrespondenzen (ginput) für alle Szenen und berechnet H + Visuals.

import os
import argparse
from pathlib import Path

# Falls dein Tk-Backend zickt, kannst du die nächsten zwei Zeilen einkommentieren:
# import matplotlib
# matplotlib.use('TkAgg')

# Imports aus deinen vorhandenen Modulen
from click_points import click_correspondences
from run_homography_pair import run_pair

ROOT = Path(__file__).resolve().parents[1]  # -> .../projects/project3
IMAGES_DIR = ROOT / "images"
OUT_A2     = ROOT / "outputs" / "a2"

SCENES = ["ihouse", "stadium_low", "stadium_high"]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def ensure_pairs(scene: str, side: str, n_points: int, reclick: bool) -> Path:
    """
    side in {'left','right'}; klickt side -> center und speichert JSON.
    Gibt Pfad zur JSON zurück.
    """
    imgA = IMAGES_DIR / f"{scene}_{side}.jpg"
    imgB = IMAGES_DIR / f"{scene}_center.jpg"
    out_json = OUT_A2 / f"{scene}_{side}_to_center.json"

    if out_json.exists() and not reclick:
        print(f"[skip click] {out_json.name} existiert – benutze vorhandene Punkte.")
        return out_json

    print(f"[click] {scene}: {side} → center  | Punkte: {n_points}")
    ensure_dir(out_json.parent)
    click_correspondences(
        imgA_path=str(imgA),
        imgB_path=str(imgB),
        n_points=n_points,
        out_json=str(out_json),
        show_numbers=True
    )
    return out_json

def compute_and_visualize(scene: str, side: str, pairs_json: Path):
    """
    Rechnet H, speichert H, RMSE und pairs.png via run_pair().
    """
    imgA = IMAGES_DIR / f"{scene}_{side}.jpg"
    imgB = IMAGES_DIR / f"{scene}_center.jpg"
    tag  = f"{scene}_{'L' if side=='left' else 'R'}_to_C"
    ensure_dir(OUT_A2)
    print(f"[H] {scene}: {side} → center  | tag={tag}")
    run_pair(
        imgA=str(imgA),
        imgB=str(imgB),
        pairs_json=str(pairs_json),
        out_dir=str(OUT_A2),
        tag=tag
    )

def main():
    ap = argparse.ArgumentParser(
        description="A.2: Klick-Korrespondenzen + Homographien für alle Szenen (left/right → center)."
    )
    ap.add_argument("--n", "--n_points", type=int, default=10, dest="n_points",
                    help="Anzahl der zu klickenden Punkte pro Paar (Default: 10).")
    ap.add_argument("--reclick", action="store_true",
                    help="Vorhandene JSONs ignorieren und neu klicken.")
    ap.add_argument("--only", type=str, default="",
                    help="Kommagetrennte Szenenliste (z. B. ihouse,stadium_low). Default: alle.")
    ap.add_argument("--pairs-only", action="store_true",
                    help="Nur klicken, keine Homographie/Visualisierung rechnen.")
    ap.add_argument("--compute-only", action="store_true",
                    help="Nur H/Visualisierung rechnen (setzt vorhandene JSONs voraus).")
    args = ap.parse_args()

    scenes = [s.strip() for s in (args.only.split(",") if args.only else SCENES) if s.strip()]
    print("[scenes]", scenes)

    # Sanity: existieren die Bilddateien?
    missing = []
    for sc in scenes:
        for name in (f"{sc}_left.jpg", f"{sc}_center.jpg", f"{sc}_right.jpg"):
            p = IMAGES_DIR / name
            if not p.exists():
                missing.append(str(p))
    if missing:
        print("\n[ERROR] folgende Bilddateien fehlen:")
        for m in missing:
            print("  -", m)
        raise SystemExit(1)

    ensure_dir(OUT_A2)

    for sc in scenes:
        print(f"\n=== Szene: {sc} ===")

        # 1) Klick JSON erzeugen (falls nötig)
        json_L = OUT_A2 / f"{sc}_left_to_center.json"
        json_R = OUT_A2 / f"{sc}_right_to_center.json"

        if not args.compute_only:
            json_L = ensure_pairs(sc, "left",  args.n_points, reclick=args.reclick)
            json_R = ensure_pairs(sc, "right", args.n_points, reclick=args.reclick)

        if args.pairs_only:
            print("[pairs-only] Überspringe H/Visualisierung.")
            continue

        # 2) H/Visualisierung (setzt JSONs voraus)
        if not json_L.exists() or not json_R.exists():
            raise SystemExit(f"[ERROR] JSON fehlt (nutze --reclick oder ohne --compute-only laufen lassen). "
                             f"Fehlt: {json_L if not json_L.exists() else json_R}")
        compute_and_visualize(sc, "left",  json_L)
        compute_and_visualize(sc, "right", json_R)

    print("\n[done] A.2: Alle gewünschten Szenen verarbeitet.")

if __name__ == "__main__":
    print(f"ROOooooooooot: {ROOT}")
    main()
