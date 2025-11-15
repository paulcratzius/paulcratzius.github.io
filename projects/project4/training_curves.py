import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("outputs/part2/lego_outputs")

metrics = np.load(out_dir / "training_metrics.npz", allow_pickle=True)
train_losses = metrics["train_losses"]              # shape (T,)
val_psnrs_arr = metrics["val_psnrs"]                # shape (K, 2) -> [step, psnr]

val_steps = val_psnrs_arr[:, 0]
val_psnrs = val_psnrs_arr[:, 1]

plt.figure(figsize=(5, 4))
plt.plot(np.arange(1, len(train_losses) + 1), train_losses)
plt.xlabel("Iteration")
plt.ylabel("MSE loss")
plt.title("Lego NeRF – Training loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "lego_train_loss.png", dpi=200)
plt.close()

plt.figure(figsize=(5, 4))
plt.plot(val_steps, val_psnrs, marker="o")
plt.xlabel("Iteration")
plt.ylabel("PSNR [dB]")
plt.title("Lego NeRF – Validation PSNR")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "lego_val_psnr.png", dpi=200)
plt.close()

print("Saved:", out_dir / "lego_train_loss.png")
print("Saved:", out_dir / "lego_val_psnr.png")
