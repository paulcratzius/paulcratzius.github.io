import numpy as np
from pathlib import Path

from train_nerf import load_nerf_npz
from rays import RaysData

PROJ_ROOT = Path(__file__).resolve().parent.parent
lego_npz = PROJ_ROOT / "inputs/part2/lego_200x200.npz"

images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K, H, W = \
    load_nerf_npz(str(lego_npz))

dataset = RaysData(images_train, K, c2ws_train)

uvs_start = 0
uvs_end   = 40_000
sample_uvs = dataset.uvs[uvs_start:uvs_end]   # shape (N,2), [x,y]

assert np.all(
    images_train[0, sample_uvs[:, 1], sample_uvs[:, 0]] ==
    dataset.pixels[uvs_start:uvs_end]
)
print("UV check passed")





import time
import numpy as np
import torch
import viser
from pathlib import Path

from train_nerf import load_nerf_npz
from rays import RaysData, sample_points_along_rays

PROJ_ROOT = Path(__file__).resolve().parent.parent
lego_npz  = PROJ_ROOT / "inputs/part2/lego_200x200.npz"

images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K, H, W = \
    load_nerf_npz(str(lego_npz))

dataset = RaysData(images_train, K, c2ws_train)

uvs_start = 0
uvs_end   = min(40_000, len(dataset.uvs))
sample_uvs = dataset.uvs[uvs_start:uvs_end]
assert np.all(
    images_train[0, sample_uvs[:, 1], sample_uvs[:, 0]] ==
    dataset.pixels[uvs_start:uvs_end]
)
print("UV check passed")

H, W = images_train.shape[1:3]
num_pixels_per_image = H * W

# indices = np.random.choice(len(dataset.rays_o), size=100, replace=False)
indices = np.random.randint(
    low=0,
    high=num_pixels_per_image,   # Bereich [0, H*W) = nur Image 0
    size=100,
)

# indices_x = np.random.randint(low=0, high=W//2, size=100)
# indices_y = np.random.randint(low=0, high=H//2, size=100)
# indices = indices_x + indices_y * W

rays_o = dataset.rays_o[indices]   # (B, 3), numpy
rays_d = dataset.rays_d[indices]   # (B, 3), numpy

rays_o_t = torch.from_numpy(rays_o).float()
rays_d_t = torch.from_numpy(rays_d).float()

points_t, _ = sample_points_along_rays(
    rays_o_t, rays_d_t,
    n_samples=32,
    near=2.0,
    far=6.0,
    perturb=True,
)
points = points_t.numpy()

server = viser.ViserServer(share=True)

for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
    server.add_camera_frustum(
        f"/cameras/{i}",
        fov=2 * np.arctan2(H / 2, K[0, 0]),
        aspect=W / H,
        scale=0.15,
        wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3],
        image=image,
    )

# Rays als Linien
for i, (o, d) in enumerate(zip(rays_o, rays_d)):
    positions = np.stack((o, o + d * 6.0))
    server.add_spline_catmull_rom(
        f"/rays/{i}", positions=positions,
    )

# Punkte entlang der Rays als Point Cloud
server.add_point_cloud(
    "/samples",
    colors=np.zeros_like(points).reshape(-1, 3),
    points=points.reshape(-1, 3),
    point_size=0.03,
)

print("Viser server running – open the share URL in your browser.")
while True:
    time.sleep(0.1)
