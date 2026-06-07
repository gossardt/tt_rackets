from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ..data import BounceDataset
from ..physics.bounce import get_surface_v
from .baselines import DerivedParameterEstimator, ParametricConstantBaseline


class ResidualMLPBaseline(DerivedParameterEstimator):
    method_name = "residual_mlp"
    deterministic = True

    def __init__(
        self,
        racket_id: str,
        random_state: int = 0,
        input_mode: str = "vs_vz",
        *,
        hidden_layer_sizes: tuple[int, ...] = (32,),
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        batch_size: int | str = "auto",
        max_iter: int = 5000,
        early_stopping: bool = True,
    ):
        self.racket_id = racket_id
        self.input_mode = input_mode
        self.base_model = ParametricConstantBaseline(racket_id=racket_id)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.alpha = float(alpha)
        self.learning_rate_init = float(learning_rate_init)
        self.batch_size = batch_size
        self.max_iter = int(max_iter)
        self.early_stopping = bool(early_stopping)
        self.residual_model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            batch_size=self.batch_size,
            early_stopping=self.early_stopping,
            max_iter=self.max_iter,
            random_state=random_state,
        )
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

    def _make_features(self, dataset: BounceDataset) -> np.ndarray:
        if self.input_mode == "full":
            return np.column_stack([dataset.v_in, dataset.w_in])
        if self.input_mode == "vs_vz":
            surface_v = get_surface_v(dataset.v_in, dataset.w_in)
            return np.column_stack([surface_v, dataset.v_in[:, 2]])
        raise ValueError(f"Unsupported input mode {self.input_mode}")

    def fit(self, dataset: BounceDataset):
        pred_v_out, pred_w_out = self.base_model.predict(dataset)
        residual_targets = np.column_stack(
            [
                dataset.v_out[:, 0] - pred_v_out[:, 0],
                dataset.v_out[:, 2] - pred_v_out[:, 2],
                dataset.w_out[:, 1] - pred_w_out[:, 1],
            ]
        )
        x_scaled = self.input_scaler.fit_transform(self._make_features(dataset))
        y_scaled = self.output_scaler.fit_transform(residual_targets)
        self.residual_model.fit(x_scaled, y_scaled)
        return self

    def predict(self, dataset: BounceDataset):
        pred_v_out, pred_w_out = self.base_model.predict(dataset)
        x_scaled = self.input_scaler.transform(self._make_features(dataset))
        residuals = self.output_scaler.inverse_transform(self.residual_model.predict(x_scaled))

        pred_v_out = pred_v_out.copy()
        pred_w_out = pred_w_out.copy()
        pred_v_out[:, 0] += residuals[:, 0]
        pred_v_out[:, 2] += residuals[:, 1]
        pred_w_out[:, 1] += residuals[:, 2]
        return pred_v_out, pred_w_out
