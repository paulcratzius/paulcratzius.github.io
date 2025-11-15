import math
import torch
import torch.nn as nn
import torch.nn.functional as F

## 2.6
import numpy as np
from pathlib import Path
from PIL import Image

from rays import RaysData, sample_points_along_rays, pixel_to_ray

print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())



class PositionalEncoding(nn.Module):
    def __init__(self, L: int):
        super().__init__()
        self.L = L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x]
        for i in range(self.L):
            freq = (2.0 ** i) * math.pi
            outs.append(torch.sin(freq * x))
            outs.append(torch.cos(freq * x))
        return torch.cat(outs, dim=-1)


class NeuralRadianceField(nn.Module):
    def __init__(
        self,
        L_xyz: int = 10,
        L_dir: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.L_xyz = L_xyz
        self.L_dir = L_dir

        self.pe_xyz = PositionalEncoding(L_xyz)
        self.pe_dir = PositionalEncoding(L_dir)

        xyz_dim = 3 * (2 * L_xyz + 1)
        dir_dim = 3 * (2 * L_dir + 1)

        self.fc1 = nn.Linear(xyz_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        self.fc5 = nn.Linear(hidden_dim + xyz_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim)

        self.sigma_fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.sigma_fc.weight)
        nn.init.constant_(self.sigma_fc.bias, 0.0)

        self.feat_fc = nn.Linear(hidden_dim, hidden_dim)
        self.rgb_fc1 = nn.Linear(hidden_dim + dir_dim, 128)
        self.rgb_fc2 = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        d = d / (torch.norm(d, dim=-1, keepdim=True) + 1e-8)

        x_enc = self.pe_xyz(x)
        d_enc = self.pe_dir(d)

        h = F.relu(self.fc1(x_enc))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))

        h = torch.cat([h, x_enc], dim=-1)

        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = F.relu(self.fc8(h))

        # --- WICHTIG: sigma mit softplus + epsilon ---
        raw_sigma = self.sigma_fc(h)
        sigma = F.softplus(raw_sigma) # so we gurantee that its never zero and kills the gradient

        feat = F.relu(self.feat_fc(h))
        h_rgb = torch.cat([feat, d_enc], dim=-1)
        h_rgb = F.relu(self.rgb_fc1(h_rgb))
        rgb = torch.sigmoid(self.rgb_fc2(h_rgb))

        return rgb, sigma

PositionalEncoding3D = PositionalEncoding
PositionalEncodingDir = PositionalEncoding
RadianceFieldNetwork = NeuralRadianceField




def volrend(sigmas: torch.Tensor,
            rgbs: torch.Tensor,
            step_size: float | torch.Tensor) -> torch.Tensor:
    device = sigmas.device
    step = torch.as_tensor(step_size, dtype=sigmas.dtype, device=device)

    sigma_delta = sigmas * step                    # (B, N, 1)
    alpha = 1.0 - torch.exp(-sigma_delta)          # (B, N, 1)

    cumsum = torch.cumsum(sigma_delta, dim=1)      # (B, N, 1)
    cumsum_prev = torch.cat(
        [torch.zeros_like(cumsum[:, :1, :]),
         cumsum[:, :-1, :]],
        dim=1
    )

    T = torch.exp(-cumsum_prev)                    # (B, N, 1)
    weights = T * alpha                            # (B, N, 1)

    colors = torch.sum(weights * rgbs, dim=1)      # (B, 3)
    return colors


## 2.6

def load_nerf_npz(npz_path: str):
    data = np.load(npz_path)

    images_train = data["images_train"].astype(np.float32) / 255.0
    c2ws_train   = data["c2ws_train"].astype(np.float32)

    images_val   = data["images_val"].astype(np.float32) / 255.0
    c2ws_val     = data["c2ws_val"].astype(np.float32)


    if "c2ws_test" in data.files:
        c2ws_test = data["c2ws_test"].astype(np.float32)
    else:
        c2ws_test = c2ws_val.copy()

    H, W = images_train.shape[1:3]
    cx, cy = W / 2.0, H / 2.0

    files = set(data.files)

    if "focal" in files:
        focal = float(data["focal"])
        K = np.array(
            [
                [focal, 0.0,  cx],
                [0.0,  focal, cy],
                [0.0,  0.0,   1.0],
            ],
            dtype=np.float32,
        )

    elif "K" in files:
        K = data["K"].astype(np.float32)
        if K.ndim == 3:
            K = K[0]

    elif "intrinsics" in files:
        K = data["intrinsics"].astype(np.float32)
        if K.ndim == 3:
            K = K[0]

    elif "Ks" in files:
        Ks = data["Ks"].astype(np.float32)
        K = Ks[0] if Ks.ndim == 3 else Ks

    else:
        raise KeyError(
            f"Could not find camera intrinsics in '{npz_path}'. "
            f"Available keys: {sorted(data.files)}"
        )

    assert K.shape == (3, 3), f"Expected K to be (3,3), got {K.shape}"
    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K, H, W




def render_image(
    model: NeuralRadianceField,
    K: np.ndarray,
    c2w: np.ndarray,
    H: int,
    W: int,
    n_samples: int,
    near: float,
    far: float,
    device: torch.device,
    chunk_rays: int = 8192,
) -> torch.Tensor:
    model.eval()

    K_t = torch.from_numpy(K).to(device)
    c2w_t = torch.from_numpy(c2w).to(device)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    uv = torch.stack([xs + 0.5, ys + 0.5], dim=-1).reshape(-1, 2)  # (N_pix,2)
    N = uv.shape[0]

    step_size = (far - near) / n_samples
    all_colors = []

    with torch.no_grad():
        for start in range(0, N, chunk_rays):
            end = min(start + chunk_rays, N)
            uv_chunk = uv[start:end]                        # (B,2)

            ray_o, ray_d = pixel_to_ray(K_t, c2w_t, uv_chunk, depth=1.0)
            pts, _ = sample_points_along_rays(
                ray_o, ray_d, n_samples, near=near, far=far, perturb=False
            )                                              # (B,S,3)

            B, S, _ = pts.shape
            pts_flat  = pts.reshape(-1, 3)                 # (B*S,3)
            dirs_flat = ray_d[:, None, :].expand(B, S, 3).reshape(-1, 3)

            rgb_s, sigma = model(pts_flat, dirs_flat)      # (B*S,3), (B*S,1)
            rgb_s  = rgb_s.view(B, S, 3)
            sigma  = sigma.view(B, S, 1)

            colors = volrend(sigma, rgb_s, step_size)      # (B,3)
            all_colors.append(colors)

    all_colors = torch.cat(all_colors, dim=0)              # (N_pix,3)
    img = all_colors.view(H, W, 3)
    return img




def save_image_torch(img: torch.Tensor, path: Path):
    """
    img: (H,W,3) in [0,1]
    """
    img_np = img.clamp(0.0, 1.0).cpu().numpy()
    img_uint8 = (img_np * 255.0).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)



def evaluate_psnr(
    model: NeuralRadianceField,
    images_val: np.ndarray,
    c2ws_val: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    n_samples: int,
    near: float,
    far: float,
    device: torch.device,
) -> float:
    model.eval()
    mse_sum = 0.0
    with torch.no_grad():
        for i in range(len(images_val)):
            gt = torch.from_numpy(images_val[i]).to(device)  # (H,W,3)
            pred = render_image(model, K, c2ws_val[i], H, W,
                                n_samples, near, far, device)
            mse = F.mse_loss(pred, gt)
            mse_sum += mse.item()
    mse_avg = mse_sum / len(images_val)
    psnr = -10.0 * math.log10(mse_avg)
    return psnr



def train_nerf_dataset(
    npz_path: str,
    out_dir: str,
    n_iters: int = 10_000,
    batch_size: int = 10_000,
    n_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    lr: float = 5e-4,
    device_str: str | None = None,
):
    if device_str is None:
        if torch.backends.mps.is_available():
            device_str = "mps"
        elif torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"

    device = torch.device(device_str)
    print(f"Using device in train_nerf_dataset: {device}")
    device = torch.device(device_str)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (images_train, c2ws_train,
     images_val, c2ws_val,
     c2ws_test, K, H, W) = load_nerf_npz(npz_path)

    rays_data = RaysData(images_train, K, c2ws_train)

    model = NeuralRadianceField(L_xyz=10, L_dir=4, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    step_size = (far - near) / n_samples

    train_losses = []
    val_psnrs = []

    for step in range(1, n_iters + 1):
        model.train()

        rays_o, rays_d, rgb = rays_data.sample_rays(batch_size, device=device)
        # rays_o, rays_d, rgb: (B,3)

        pts, _ = sample_points_along_rays(
            rays_o, rays_d, n_samples, near=near, far=far, perturb=True
        )  # (B,S,3)
        B, S, _ = pts.shape

        pts_flat  = pts.reshape(-1, 3)                     # (B*S,3)
        dirs_flat = rays_d[:, None, :].expand(B, S, 3).reshape(-1, 3)

        rgb_s, sigma = model(pts_flat, dirs_flat)          # (B*S,3), (B*S,1)
        rgb_s = rgb_s.view(B, S, 3)
        sigma = sigma.view(B, S, 1)

        pred_colors = volrend(sigma, rgb_s, step_size)     # (B,3)

        loss = F.mse_loss(pred_colors, rgb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if step % 50 == 0:
            psnr_train = -10.0 * math.log10(loss.item())
            print(f"[{step:05d}/{n_iters}] loss={loss.item():.4f}, "
                  f"train-psnr≈{psnr_train:.2f} dB")
            
        if step % 500 == 0 or step == n_iters:
            psnr_val = evaluate_psnr(
                model, images_val, c2ws_val, K, H, W,
                n_samples, near, far, device
            )
            val_psnrs.append((step, psnr_val))
            print(f"  -> val PSNR = {psnr_val:.2f} dB")

            pred_img = render_image(
                model, K, c2ws_val[0], H, W,
                n_samples, near, far, device
            )
            save_image_torch(pred_img, out_dir / f"val_step{step:05d}.png")

    np.savez(
        out_dir / "training_metrics.npz",
        train_losses=np.array(train_losses),
        val_psnrs=np.array(val_psnrs),
    )

    torch.save(model.state_dict(), out_dir / "nerf_model.pt")

    return model, (images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K, H, W)



def look_at_origin(pos: np.ndarray) -> np.ndarray:
    forward = -pos / np.linalg.norm(pos)
    up = np.array([0., 1., 0.], dtype=np.float32)

    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos
    return c2w

def rot_x(phi: float) -> np.ndarray:
    return np.array([
        [math.cos(phi), -math.sin(phi), 0., 0.],
        [math.sin(phi),  math.cos(phi), 0., 0.],
        [0.,             0.,            1., 0.],
        [0.,             0.,            0., 1.],
    ], dtype=np.float32)


import imageio.v2 as imageio

def render_circular_gif(
    model: NeuralRadianceField,
    K: np.ndarray,
    H: int,
    W: int,
    out_path: str,
    n_samples: int,
    near: float,
    far: float,
    start_pos: np.ndarray = np.array([1., 0., 0.], dtype=np.float32),
    num_frames: int = 60,
    device_str: str | None = None,
):
    if device_str is None:
        if torch.backends.mps.is_available():
            device_str = "mps"
        elif torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"

    device = torch.device(device_str)
    print(f"Using device in render_circular_gif: {device}")

    frames = []

    for phi in np.linspace(360., 0., num_frames, endpoint=False):
        c2w = look_at_origin(start_pos)
        extrinsic = rot_x(phi / 180.0 * math.pi) @ c2w

        img_t = render_image(
            model, K, extrinsic, H, W,
            n_samples=n_samples, near=near, far=far, device=device,
        )
        img_np = (img_t.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        frames.append(img_np)

    imageio.mimsave(out_path, frames, fps=15)
    print(f"Saved GIF to {out_path}")



if __name__ == "__main__":
    torch.manual_seed(42)

    PROJ_ROOT = Path(__file__).resolve().parent.parent
    lego_npz  = PROJ_ROOT / "inputs/part2/lego_200x200.npz"

    # Train Lego
    model, data = train_nerf_dataset(
        npz_path=str(lego_npz),
        out_dir=str(PROJ_ROOT / "outputs/lego_nerf"),
        n_iters=10000,
        batch_size=4096,
        n_samples=64,
        near=2.0,
        far=6.0,
        lr=5e-4,
        device_str=None,  # nimmt automatisch MPS/GPU/CPU
    )

