from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..paths import ORIGINAL_DATA_DIR, TEST_DATA_DIR, TRAIN_DATA_DIR

SPLIT_DIRS = {
    "original": ORIGINAL_DATA_DIR,
    "train": TRAIN_DATA_DIR,
    "test": TEST_DATA_DIR,
}


@dataclass(frozen=True)
class BounceDataset:
    v_in: np.ndarray
    w_in: np.ndarray
    v_out: np.ndarray
    w_out: np.ndarray
    raw: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.v_in.shape[0])


def preprocess_data(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v_in = data[:, 8:11]
    w_in = data[:, 2:5]
    v_out = data[:, 11:14]
    w_out = data[:, 5:8]
    return v_in, w_in, v_out, w_out


def load_csv_data(csv_path: str | Path) -> np.ndarray:
    return np.genfromtxt(Path(csv_path), dtype=float, delimiter=",", skip_header=1)


def resolve_racket_csv_path(
    racket_id: str | int,
    split: str = "original",
    data_dir: str | Path | None = None,
) -> Path:
    if data_dir is not None:
        root = Path(data_dir)
    else:
        try:
            root = SPLIT_DIRS[split]
        except KeyError as exc:
            raise ValueError(f"Unknown split {split}") from exc
    racket_suffix = f"{int(racket_id):02d}" if isinstance(racket_id, int) or str(racket_id).isdigit() else str(racket_id)
    candidates = [root / f"racket_{racket_suffix}.csv"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No CSV found for racket {racket_suffix} in {root}")


def load_bounce_dataset(csv_path: str | Path) -> BounceDataset:
    data = load_csv_data(csv_path)
    v_in, w_in, v_out, w_out = preprocess_data(data)
    return BounceDataset(v_in=v_in, w_in=w_in, v_out=v_out, w_out=w_out, raw=data)


def subset_bounce_dataset(dataset: BounceDataset, indices: np.ndarray) -> BounceDataset:
    raw = None if dataset.raw is None else dataset.raw[indices]
    return BounceDataset(
        v_in=dataset.v_in[indices],
        w_in=dataset.w_in[indices],
        v_out=dataset.v_out[indices],
        w_out=dataset.w_out[indices],
        raw=raw,
    )


def load_train_test_datasets(racket_id: str | int) -> tuple[BounceDataset, BounceDataset]:
    train_dataset = load_bounce_dataset(resolve_racket_csv_path(racket_id, split="train"))
    test_dataset = load_bounce_dataset(resolve_racket_csv_path(racket_id, split="test"))
    return train_dataset, test_dataset
