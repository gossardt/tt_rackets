from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"

ORIGINAL_DATA_DIR = DATA_ROOT / "original"
TRAIN_DATA_DIR = DATA_ROOT / "train"
TEST_DATA_DIR = DATA_ROOT / "test"

DEFAULT_PLOTS_DIR = RESULTS_ROOT / "plots"
DEFAULT_FULL_BENCHMARK_DIR = RESULTS_ROOT / "full_benchmark"
RACKETS_CONFIG_PATH = PYTHON_ROOT / "racket_dynamics" / "configs" / "rackets.py"


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path
