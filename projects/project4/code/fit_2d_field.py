import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============ CONFIG ============
# Pfad zu deinem Bild (Fuchs); relativ zum project4-Ordner
IMAGE_RELPATH = "inputs/part1/monroe.jpg"

# Hyperparameter (später für Sweeps einfach ändern und Script erneut starten)
MAX_FREQ_L = 10          # Positional Encoding L
WIDTH = 256              # Hidden width
DEPTH = 4                # Anzahl Hidden-Layer
LR = 1e-2                # Learning rate
NUM_ITERS = 3000         # Trainingsschritte
BATCH_SIZE = 10_000      # # zufällig gesampelte Pixel pro Schritt
EVAL_EVERY = 100         # alle wieviel Schritte komplette Rekonstruktion + PSNR


# ============ Positional Encoding ============
class PositionalEncoding(nn.Module):
    """
    Sinusoidal PE wie im NeRF-Paper (siehe Eq. 4).
    Input:  (B, 2) mit Werten in [0,1]
    Output: (B, 2 + 4L)
    """
    def __init__(self, L: int):
        super().__init__()
        self.L = L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x]  # Originalkoordinaten behalten
        for i in range(self.L):
            freq = (2.0 ** i) * math.pi
            outs.append(torch.sin(freq * x))
            outs.append(torch.cos(freq * x))
        return torch.cat(outs, dim=-1)


# ============ MLP ============
class NeuralField2D(nn.Module):
    """
    Einfaches MLP mit ReLU + Sigmoid am Ende.
    """
    def __init__(self, in_dim: int, width: int = 256, depth: int = 4):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, width))
            layers.append(nn.ReLU(inplace=True))
            dim = width
        layers.append(nn.Linear(width, 3))
        layers.append(nn.Sigmoid())  # Output in [0,1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============ Hilfsfunktionen ============
def load_image_and_coords(image_path: Path, device: torch.device):
    """
    Lädt ein Bild und erzeugt:
    - coords: (N,2) normierte Pixelkoordinaten in [0,1]
    - colors: (N,3) RGB in [0,1]
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0  # (H,W,3)
    img_t = torch.from_numpy(img_np)  # (H,W,3)
    H, W, _ = img_t.shape

    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    # Normierung auf [0,1]
    coords = torch.stack([xs / W, ys / H], dim=-1)  # (H,W,2)
    coords = coords.reshape(-1, 2).to(device)
    colors = img_t.reshape(-1, 3).to(device)

    return coords, colors, H, W


def psnr_from_mse(mse_tensor: torch.Tensor) -> torch.Tensor:
    """
    PSNR bei Bildern in [0,1], wie in der Aufgabenstellung:
    PSNR = 10 * log10(1 / MSE)
    """
    return 10.0 * torch.log10(1.0 / mse_tensor)


def render_full_image(model, pe, coords, H, W, device):
    """
    Rendert das komplette Bild aus dem aktuellen Modell.
    """
    model.eval()
    with torch.no_grad():
        x = coords.to(device)
        x_pe = pe(x)
        preds = model(x_pe)  # (N,3)
        img_recon = preds.view(H, W, 3).cpu().numpy()
    return img_recon  # float in [0,1]


def save_image(img_np, path: Path):
    """
    Speichert ein (H,W,3)-Array mit Werten in [0,1] als PNG.
    """
    img_clipped = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_clipped).save(path)


# ============ Haupt-Trainingsroutine ============
def train_single_image(
    image_path: Path,
    out_dir: Path,
    L: int = MAX_FREQ_L,
    width: int = WIDTH,
    depth: int = DEPTH,
    lr: float = LR,
    num_iters: int = NUM_ITERS,
    batch_size: int = BATCH_SIZE,
    eval_every: int = EVAL_EVERY,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Daten laden
    coords, colors, H, W = load_image_and_coords(image_path, device)
    N = coords.shape[0]
    batch_size = min(batch_size, N)

    print(f"Loaded image {image_path.name} with size {W}x{H}, N={N} pixels.")
    print(f"Positional Encoding L={L} => input dim = {2 + 4 * L}")
    print(f"MLP: depth={depth}, width={width}, lr={lr}")

    # 2) Modell + PE
    pe = PositionalEncoding(L=L).to(device)
    in_dim = 2 + 4 * L
    model = NeuralField2D(in_dim=in_dim, width=width, depth=depth).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    steps_list = []
    psnr_list = []

    # Anfangsrekonstruktion (Step 0)
    img_recon0 = render_full_image(model, pe, coords, H, W, device)
    save_image(img_recon0, out_dir / "recon_step0000.png")

    for step in range(1, num_iters + 1):
        model.train()

        # --- zufällige Pixel samplen ---
        idx = torch.randint(0, N, (batch_size,), device=device)
        coords_b = coords[idx]          # (B,2)
        colors_b = colors[idx]          # (B,3)

        # --- Forward ---
        preds = model(pe(coords_b))
        loss = criterion(preds, colors_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Logging / Auswertung ---
        if step % 50 == 0:
            print(f"[{step:04d}/{num_iters}] loss = {loss.item():.6f}")

        # volle Rekonstruktion bei 10, 50, allen 100er-Schritten und am Ende
        if step in (10, 50) or (step % eval_every == 0) or (step == num_iters):
            model.eval()
            with torch.no_grad():
                # vollständige Rekonstruktion + MSE/PSNR über alle Pixel
                preds_full = model(pe(coords))
                mse_full = criterion(preds_full, colors)
                psnr = psnr_from_mse(mse_full).item()

            steps_list.append(step)
            psnr_list.append(psnr)
            print(f"  -> full-image MSE={mse_full.item():.6f}, PSNR={psnr:.2f} dB")

            # Bild speichern
            img_recon = preds_full.view(H, W, 3).detach().cpu().numpy()
            save_image(img_recon, out_dir / f"recon_step{step:04d}.png")

    # --- PSNR-Kurve als Plot speichern ---
    if len(steps_list) > 0:
        plt.figure()
        plt.plot(steps_list, psnr_list, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("PSNR (dB)")
        plt.title(f"PSNR vs Iteration ({image_path.name})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "psnr_curve.png")
        plt.close()

    # --- PSNR-Werte auch roh speichern ---
    np.savez(out_dir / "metrics_psnr.npz",
             steps=np.array(steps_list),
             psnr=np.array(psnr_list))

    # Architektur / Settings in Textform für Report
    with open(out_dir / "config.txt", "w") as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"H={H}, W={W}, N={N}\n")
        f.write(f"L={L}, in_dim={in_dim}\n")
        f.write(f"width={width}, depth={depth}\n")
        f.write(f"lr={lr}, num_iters={num_iters}, batch_size={batch_size}\n")


# ============ Main ============
if __name__ == "__main__":
    # project4 root bestimmen
    THIS_DIR = Path(__file__).resolve().parent
    PROJ_ROOT = THIS_DIR.parent

    image_path = PROJ_ROOT / IMAGE_RELPATH

    out_dir = PROJ_ROOT / "outputs" / "part1" / (
        f"monroe_L{MAX_FREQ_L}_W{WIDTH}"
    )

    train_single_image(
        image_path=image_path,
        out_dir=out_dir,
        L=MAX_FREQ_L,
        width=WIDTH,
        depth=DEPTH,
        lr=LR,
        num_iters=NUM_ITERS,
        batch_size=BATCH_SIZE,
        eval_every=EVAL_EVERY,
    )
