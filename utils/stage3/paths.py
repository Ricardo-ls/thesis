from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLEAN_ABS_INPUT_PATH = ROOT / "datasets" / "processed" / "data_eth_ucy_20.npy"

OUTPUT_DIR = ROOT / "outputs" / "stage3"
PHASE1_OUTPUT_DIR = OUTPUT_DIR / "phase1" / "canonical_room3"
DATA_OUT_DIR = PHASE1_OUTPUT_DIR / "data"
BASELINE_OUT_DIR = PHASE1_OUTPUT_DIR / "baselines"
EVAL_OUT_DIR = PHASE1_OUTPUT_DIR / "eval"
LOG_OUT_DIR = PHASE1_OUTPUT_DIR / "logs"
DOCS_DIR = ROOT / "docs" / "stage3"

EXPERIMENTS_DIR = DATA_OUT_DIR / "experiments"
CLEAN_ROOM3_PATH = DATA_OUT_DIR / "clean_windows_room3.npz"
CLEAN_ROOM3_META_PATH = DATA_OUT_DIR / "clean_windows_room3_meta.json"
OCCUPANCY_ROOM3_EMPTY_PATH = DATA_OUT_DIR / "occupancy_map_room3_empty.npz"

DEFAULT_EXPERIMENT_ID = "span20_fixed_seed42"
LINEAR_METHOD_TAG = "linear_interp"
SAVGOL_METHOD_TAG = "savgol_w5_p2"
KALMAN_METHOD_TAG = "kalman_cv_dt1.0_q1e-3_r1e-2"


def experiment_data_dir(experiment_id: str):
    return EXPERIMENTS_DIR / experiment_id


def missing_span_path(experiment_id: str):
    return experiment_data_dir(experiment_id) / "missing_span_windows.npz"


def baseline_method_dir(experiment_id: str, method_tag: str):
    return BASELINE_OUT_DIR / experiment_id / method_tag


def baseline_results_path(experiment_id: str, method_tag: str):
    return baseline_method_dir(experiment_id, method_tag) / "results.npz"


def eval_method_dir(experiment_id: str, method_tag: str):
    return EVAL_OUT_DIR / experiment_id / method_tag


def reconstruction_metrics_path(experiment_id: str, method_tag: str):
    return eval_method_dir(experiment_id, method_tag) / "reconstruction_metrics.json"


def geometry_metrics_path(experiment_id: str, method_tag: str):
    return eval_method_dir(experiment_id, method_tag) / "geometry_metrics.json"


def ensure_stage3_dirs():
    dirs = {
        "output": OUTPUT_DIR,
        "phase1": PHASE1_OUTPUT_DIR,
        "data": DATA_OUT_DIR,
        "experiments": EXPERIMENTS_DIR,
        "baselines": BASELINE_OUT_DIR,
        "eval": EVAL_OUT_DIR,
        "logs": LOG_OUT_DIR,
        "docs": DOCS_DIR,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
