# demo_edges.py
import os, cv2, numpy as np
from solution import load_image
from pyramid import edges_prewitt   # oder edges_central_diff

if __name__ == "__main__":
    b, g, r = load_image("projects/project1/images/large/emir.tif")
    edge = edges_prewitt(b)  # float32 ~N(0,1)

    # normieren → [0,255] fürs Speichern
    vis = (np.clip(edge - edge.min(), 0, None) / (edge.max()-edge.min()+1e-8) * 255).astype(np.uint8)

    out_dir = "projects/project1/outputs/demos"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "emir_edges.jpg")
    cv2.imwrite(out_path, vis)
    print("edge vis:", out_path)
