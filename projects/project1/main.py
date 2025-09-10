from pathlib import Path
from runner import run_batch
from solution import process_one

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# project root
ROOT = Path(__file__).resolve().parent

# input / output folders
IN_SMALL  = ROOT / "images" / "small"
OUT_SMALL = ROOT / "outputs" / "rgb_small"

IN_LARGE  = ROOT / "images" / "large"
OUT_LARGE = ROOT / "outputs" / "rgb_large"

IN_EXTRA  = ROOT / "images" / "large-extra"
OUT_EXTRA = ROOT / "outputs" / "rgb_large_extra"


if __name__ == "__main__":
    
    
    # small
    print("[info] small in  =", IN_SMALL.resolve())
    print("[info] small out =", OUT_SMALL.resolve())
    ensure_dir(OUT_SMALL)
    run_batch(
        str(IN_SMALL), str(OUT_SMALL),
        process_one_func=process_one,
        methods=("single-ssd","single-ncc","phase"),
        csv_log="results.csv",
    )

    # large
    print("[info] large in  =", IN_LARGE.resolve())
    print("[info] large out =", OUT_LARGE.resolve())
    ensure_dir(OUT_LARGE)
    run_batch(
        str(IN_LARGE), str(OUT_LARGE),
        process_one_func=process_one,
        methods=("single-ssd", "pyramid-ssd", "pyramid-ncc", "phase"),  # ← added 'single-ssd' first
        csv_log="results.csv",
    )

    # extra large
    print("[info] extra in  =", IN_EXTRA.resolve())
    print("[info] extra out =", OUT_EXTRA.resolve())
    ensure_dir(OUT_EXTRA)
    run_batch(
        str(IN_EXTRA), str(OUT_EXTRA),
        process_one_func=process_one,
        methods=("single-ssd", "pyramid-ssd", "pyramid-ncc", "phase"),  # ← added 'single-ssd' first
        csv_log="results.csv",
    )
