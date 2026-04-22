from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLEAN_ABS_INPUT_PATH = ROOT / "datasets" / "processed" / "data_eth_ucy_20.npy"

OUTPUT_DIR = ROOT / "outputs" / "stage3"
DATA_OUT_DIR = OUTPUT_DIR / "data"
BASELINE_OUT_DIR = OUTPUT_DIR / "baselines"
EVAL_OUT_DIR = OUTPUT_DIR / "eval"
DOCS_DIR = ROOT / "docs" / "stage3"


def ensure_stage3_dirs():
    dirs = {
        "output": OUTPUT_DIR,
        "data": DATA_OUT_DIR,
        "baselines": BASELINE_OUT_DIR,
        "eval": EVAL_OUT_DIR,
        "docs": DOCS_DIR,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
