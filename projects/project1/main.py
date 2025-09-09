# projects/project1/main.py
from pathlib import Path
from runner import run_batch
import solution   # damit wir solution.process_one übergeben können

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    IN_ROOT = BASE / "images"
    OUT_ROOT = BASE / "outputs"

    IN_SMALL  = IN_ROOT / "small"
    IN_LARGE  = IN_ROOT / "large"
    OUT_SMALL = OUT_ROOT / "rgb_small"
    OUT_LARGE = OUT_ROOT / "rgb_large"

    if IN_SMALL.exists():
        print(f"[info] small in  = {IN_SMALL}")
        print(f"[info] small out = {OUT_SMALL}")
        run_batch(
            in_dir="projects/project1/images/small",
            out_dir="projects/project1/outputs/rgb_small",
            methods=('single-ssd','single-ncc','phase'),   # phase optional
            process_one_func=solution.process_one)

    if IN_LARGE.exists():
        print(f"[info] large in  = {IN_LARGE}")
        print(f"[info] large out = {OUT_LARGE}")
        run_batch(str(IN_LARGE), str(OUT_LARGE),
                  methods=("pyramid-ssd","pyramid-ncc","phase"),
                  process_one_func=solution.process_one)
