import torch

def transform(T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(x[..., :1])
    x_h = torch.cat([x, ones], dim=-1)          # (..., 4)
    x_h_col = x_h.unsqueeze(-1)                 # (..., 4, 1)

    x_out_h = T @ x_h_col                       # (..., 4, 1)
    x_out_h = x_out_h.squeeze(-1)              # (..., 4)

    x_out = x_out_h[..., :3] / x_out_h[..., 3:].clamp(min=1e-8)
    return x_out


def pixel_to_camera(
    K: torch.Tensor,
    uv: torch.Tensor,
    s: torch.Tensor | float,
) -> torch.Tensor:
    
    if not torch.is_tensor(uv):
        uv = torch.tensor(uv, dtype=K.dtype, device=K.device)

    ones = torch.ones_like(uv[..., :1])
    uv1 = torch.cat([uv, ones], dim=-1)  # (..., 3)

    if not torch.is_tensor(s):
        s = torch.tensor(s, dtype=K.dtype, device=K.device)
    s = s.to(uv1.device).to(uv1.dtype)
    
    if s.ndim == 0:
        s = s * torch.ones_like(uv1[..., :1])       # (..., 1)
    elif s.shape == uv1.shape[:-1]:
        s = s.unsqueeze(-1)                         # (..., 1)

    K_inv = torch.inverse(K)
    xyz_c = (s * uv1) @ K_inv.transpose(-1, -2)     # (..., 3)

    return xyz_c



def pixel_to_ray(
    K: torch.Tensor,
    c2w: torch.Tensor,      # (..., 4, 4)
    uv: torch.Tensor,       # (..., 2)
    depth: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = c2w.device
    K = K.to(device)
    uv = uv.to(device)

    ray_o = c2w[..., :3, 3]              # (..., 3)

    X_c = pixel_to_camera(K, uv, depth)  # (..., 3)

    X_w = transform(c2w, X_c)            # (..., 3)

    ray_o_exp = ray_o
    while ray_o_exp.dim() < X_w.dim():
        ray_o_exp = ray_o_exp.unsqueeze(0)
    ray_o_exp = ray_o_exp.expand_as(X_w)

    ray_d = X_w - ray_o_exp
    ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True).clamp(min=1e-8)

    return ray_o_exp, ray_d



def sample_random_rays(
    images: torch.Tensor,   # (N_imgs, H, W, 3), float in [0,1]
    c2ws: torch.Tensor,     # (N_imgs, 4, 4)
    K: torch.Tensor,        # (3, 3)
    num_rays: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    device = images.device
    K = K.to(device)
    c2ws = c2ws.to(device)

    N_imgs, H, W, _ = images.shape

    img_ids = torch.randint(0, N_imgs, (num_rays,), device=device)

    u_int = torch.randint(0, W, (num_rays,), device=device)
    v_int = torch.randint(0, H, (num_rays,), device=device)

    u = u_int.float() + 0.5
    v = v_int.float() + 0.5
    uv = torch.stack([u, v], dim=-1)  # (num_rays, 2)

    rgb = images[img_ids, v_int, u_int, :]  # (num_rays, 3)

    c2w_rays = c2ws[img_ids]  # (num_rays, 4, 4)

    ray_o, ray_d = pixel_to_ray(K, c2w_rays, uv, depth=1.0)

    return ray_o, ray_d, rgb, img_ids, uv


def sample_points_along_rays(
    ray_o: torch.Tensor,   # (N_rays, 3)
    ray_d: torch.Tensor,   # (N_rays, 3), normiert
    n_samples: int,
    near: float = 2.0,
    far: float = 6.0,
    perturb: bool = True,
):
    device = ray_o.device
    N_rays = ray_o.shape[0]

    t_edges = torch.linspace(near, far, n_samples + 1,
                             device=device)          # (n_samples+1,)
    t_lower = t_edges[:-1]                           # (n_samples,)
    t_upper = t_edges[1:]                            # (n_samples,)

    if perturb:
        t_rand = torch.rand(N_rays, n_samples, device=device)
        t = t_lower + (t_upper - t_lower) * t_rand   # (N_rays, n_samples)
    else:
        t_mid = 0.5 * (t_lower + t_upper)            # (n_samples,)
        t = t_mid.expand(N_rays, n_samples)          # (N_rays, n_samples)

    ray_o_exp = ray_o.unsqueeze(1)                   # (N_rays, 1, 3)
    ray_d_exp = ray_d.unsqueeze(1)                   # (N_rays, 1, 3)
    pts = ray_o_exp + ray_d_exp * t.unsqueeze(-1)    # (N_rays, n_samples, 3)

    return pts, t



import numpy as np

def sample_along_rays(
    rays_o: torch.Tensor | np.ndarray,
    rays_d: torch.Tensor | np.ndarray,
    n_samples: int = 64,
    near: float = 2.0,
    far: float = 6.0,
    random: bool = True,
    perturb: bool | None = None,
):
    if perturb is not None:
        random = perturb

    # Eingabetyp merken
    input_is_numpy = isinstance(rays_o, np.ndarray)

    if input_is_numpy:
        r_o = torch.from_numpy(rays_o.astype(np.float32))
        r_d = torch.from_numpy(rays_d.astype(np.float32))
    else:
        r_o = rays_o
        r_d = rays_d

    pts, t = sample_points_along_rays(
        ray_o=r_o,
        ray_d=r_d,
        n_samples=n_samples,
        near=near,
        far=far,
        perturb=random,
    )

    if input_is_numpy:
        return pts.detach().cpu().numpy()
    else:
        return pts
class RaysData:
    def __init__(self, images: np.ndarray, K: np.ndarray, c2ws: np.ndarray):
        assert images.ndim == 4, "images muss (N, H, W, 3) sein"
        assert K.shape == (3, 3)
        assert c2ws.ndim == 3 and c2ws.shape[1:] == (4, 4)

        self.images = images.astype(np.float32)
        self.K_np = K.astype(np.float32)
        self.c2ws_np = c2ws.astype(np.float32)

        N_imgs, H, W, _ = self.images.shape
        self.N_imgs, self.H, self.W = N_imgs, H, W

        xs = np.arange(W, dtype=np.int32)
        ys = np.arange(H, dtype=np.int32)
        # indexing='xy': xv = x, yv = y
        xv, yv = np.meshgrid(xs, ys, indexing="xy")  # (H, W)
        uvs_single = np.stack([xv, yv], axis=-1).reshape(-1, 2)  # (H*W, 2), int

        self.uvs = np.concatenate(
            [uvs_single for _ in range(N_imgs)], axis=0
        )  # (N_imgs*H*W, 2), int32

        self.pixels = self.images.reshape(-1, 3)  # (N_imgs*H*W, 3), float32

        K_t = torch.from_numpy(self.K_np)  # (3,3), float32
        uvs_centers = uvs_single.astype(np.float32) + 0.5  # Pixelmitten
        uv_t = torch.from_numpy(uvs_centers)               # (H*W, 2)

        rays_o_list = []
        rays_d_list = []

        for i in range(N_imgs):
            c2w_i = torch.from_numpy(self.c2ws_np[i])  # (4,4)
            # pixel_to_ray erwartet torch-Tensoren
            ray_o_i, ray_d_i = pixel_to_ray(K_t, c2w_i, uv_t, depth=1.0)
            # -> (H*W, 3)
            rays_o_list.append(ray_o_i.detach().cpu().numpy())
            rays_d_list.append(ray_d_i.detach().cpu().numpy())

        self.rays_o = np.concatenate(rays_o_list, axis=0).astype(np.float32)
        self.rays_d = np.concatenate(rays_d_list, axis=0).astype(np.float32)

        assert self.rays_o.shape[0] == self.pixels.shape[0] == self.uvs.shape[0]
        self.num_rays = self.rays_o.shape[0]

    def __len__(self):
        return self.num_rays

    def sample_rays(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ):
        idx = np.random.randint(0, self.num_rays, size=batch_size)

        rays_o = torch.from_numpy(self.rays_o[idx]).to(device)
        rays_d = torch.from_numpy(self.rays_d[idx]).to(device)
        pixels = torch.from_numpy(self.pixels[idx]).to(device)

        return rays_o, rays_d, pixels




if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(0)

    # Test 1: transform + Inverse
    R, _ = torch.linalg.qr(torch.randn(3, 3))
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = torch.tensor([0.3, -0.2, 1.5])
    T_inv = torch.inverse(T)

    pts_world = torch.randn(100, 3)
    pts_cam = transform(T_inv, pts_world)
    pts_world_rec = transform(T, pts_cam)

    assert torch.allclose(pts_world_rec, pts_world, atol=1e-5), "transform/T_inv Test failed"

    # Test 2: pixel_to_camera entspricht Projektion
    H, W = 200, 200
    fx, fy = 100.0, 100.0
    cx, cy = W / 2, H / 2
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=torch.float32)

    uv = torch.tensor([[cx + 10, cy + 5],
                       [cx - 20, cy - 15]], dtype=torch.float32)
    s = torch.tensor([2.0, 4.0])
    xyz_c = pixel_to_camera(K, uv, s)

    proj = (K @ xyz_c.T).T          # (N, 3)
    uv_rec = proj[:, :2] / proj[:, 2:]
    assert torch.allclose(uv_rec, uv, atol=1e-4), "pixel_to_camera Test failed"

    # Test 3: pixel_to_ray Normierung
    c2w = torch.eye(4)
    ray_o, ray_d = pixel_to_ray(K, c2w, uv, depth=1.0)
    norms = torch.norm(ray_d, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "pixel_to_ray normalization failed"

    # Test 4: sample_points_along_rays Range/Shape
    pts, t = sample_points_along_rays(ray_o, ray_d, n_samples=64, near=2.0, far=6.0, perturb=False)
    assert pts.shape == (uv.shape[0], 64, 3), "pts shape wrong"
    assert torch.all(t >= 2.0 - 1e-6) and torch.all(t <= 6.0 + 1e-6), "t outside [near, far]"

    # Test 5: RaysData uvs/pixels Align
    N_imgs, H_small, W_small = 1, 5, 4
    images_small = np.random.rand(N_imgs, H_small, W_small, 3).astype(np.float32)
    K_small = np.array([[50.0, 0.0, W_small / 2],
                        [0.0, 50.0, H_small / 2],
                        [0.0, 0.0, 1.0]], dtype=np.float32)
    c2ws_small = np.tile(np.eye(4, dtype=np.float32)[None, ...], (N_imgs, 1, 1))

    ds = RaysData(images_small, K_small, c2ws_small)
    sample_uvs = ds.uvs[:H_small * W_small]
    assert np.allclose(images_small[0, sample_uvs[:, 1], sample_uvs[:, 0]],
                       ds.pixels[:H_small * W_small]), "RaysData uvs/pixels mismatch"

    print("All tests in rays.py passed")
