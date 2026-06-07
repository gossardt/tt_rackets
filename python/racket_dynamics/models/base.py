from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterPrediction:
    cor: np.ndarray
    cor_std: np.ndarray | None = None
    aux: np.ndarray | None = None
    aux_std: np.ndarray | None = None
    aux_name: str = "alpha"


class Estimator(ABC):
    method_name = "estimator"
    task_type = "unspecified"

    @abstractmethod
    def fit(self, train_data):
        raise NotImplementedError


class DirectBounceEstimator(Estimator):
    task_type = "direct"

    @abstractmethod
    def predict(self, v_in, w_in):
        raise NotImplementedError

    def predict_uncertainty(self, v_in, w_in):
        return None


class ParameterEstimator(ABC):
    @abstractmethod
    def predict_parameters(self, dataset) -> ParameterPrediction:
        raise NotImplementedError
