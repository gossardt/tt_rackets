from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..data import BounceDataset
from ..models.base import ParameterPrediction
from .metrics import BounceMetrics


@dataclass(frozen=True)
class RacketReport:
    racket_id: str
    num_train: int
    num_test: int
    metrics: BounceMetrics


@dataclass(frozen=True)
class ExperimentReport:
    method_name: str
    split_name: str
    rackets: list[RacketReport]
    mean_velocity_mae: float
    std_velocity_mae: float
    mean_spin_mae: float
    std_spin_mae: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RacketEvaluation:
    racket_id: str
    train_dataset: BounceDataset
    test_dataset: BounceDataset
    pred_v_out: np.ndarray
    pred_w_out: np.ndarray
    train_parameters: ParameterPrediction | None
    report: RacketReport


@dataclass(frozen=True)
class ExperimentResult:
    report: ExperimentReport
    evaluations: list[RacketEvaluation]
