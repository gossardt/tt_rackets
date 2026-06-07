from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ..configs import RACKET_PARAMS
from ..data import BounceDataset
from ..physics.bounce import est_alpha, get_cor, get_kp, get_surface_v, lin_bounce
from ..physics.constants import M
from .base import ParameterEstimator, ParameterPrediction
from .baselines import BounceMethod


class MLPParameterBaseline(BounceMethod, ParameterEstimator):
    method_name = "mlp_parameters"
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
        self.friction_model = RACKET_PARAMS[racket_id]["friction_model"]
        self.input_mode = input_mode
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.alpha = float(alpha)
        self.learning_rate_init = float(learning_rate_init)
        self.batch_size = batch_size
        self.max_iter = int(max_iter)
        self.early_stopping = bool(early_stopping)
        self.cor_model = MLPRegressor(
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
        self.aux_model = MLPRegressor(
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
        self.cor_scaler = StandardScaler()
        self.aux_scaler = StandardScaler()

    def _make_features(self, dataset: BounceDataset) -> np.ndarray:
        if self.input_mode == "full":
            return np.column_stack([dataset.v_in, dataset.w_in])
        if self.input_mode == "vs_vz":
            surface_v = get_surface_v(dataset.v_in, dataset.w_in)
            return np.column_stack([surface_v, dataset.v_in[:, 2]])
        raise ValueError(f"Unsupported input mode {self.input_mode}")

    def fit(self, dataset: BounceDataset):
        x = self._make_features(dataset)
        x_scaled = self.input_scaler.fit_transform(x)

        cor = get_cor(dataset.v_in, dataset.v_out)
        cor_scaled = self.cor_scaler.fit_transform(cor.reshape(-1, 1)).ravel()
        self.cor_model.fit(x_scaled, cor_scaled)

        if self.friction_model == "coulomb":
            aux_targets = np.mean(
                np.stack(est_alpha(dataset.v_in, dataset.w_in, dataset.v_out, dataset.w_out), axis=0),
                axis=0,
            )
        else:
            aux_targets = np.mean(
                np.stack(get_kp(dataset.v_in, dataset.w_in, dataset.v_out, dataset.w_out), axis=0),
                axis=0,
            )
        aux_scaled = self.aux_scaler.fit_transform(aux_targets.reshape(-1, 1)).ravel()
        self.aux_model.fit(x_scaled, aux_scaled)
        return self

    def predict(self, dataset: BounceDataset):
        x = self._make_features(dataset)
        x_scaled = self.input_scaler.transform(x)
        pred_cor = self.cor_scaler.inverse_transform(self.cor_model.predict(x_scaled).reshape(-1, 1)).ravel()
        pred_aux = self.aux_scaler.inverse_transform(self.aux_model.predict(x_scaled).reshape(-1, 1)).ravel()
        pred_alpha = pred_aux if self.friction_model == "coulomb" else pred_aux / M
        return lin_bounce(dataset.v_in, dataset.w_in, pred_cor, pred_alpha)

    def predict_parameters(self, dataset: BounceDataset):
        x = self._make_features(dataset)
        x_scaled = self.input_scaler.transform(x)
        pred_cor = self.cor_scaler.inverse_transform(self.cor_model.predict(x_scaled).reshape(-1, 1)).ravel()
        pred_aux = self.aux_scaler.inverse_transform(self.aux_model.predict(x_scaled).reshape(-1, 1)).ravel()
        aux_name = "alpha" if self.friction_model == "coulomb" else "kp"
        return ParameterPrediction(cor=pred_cor, aux=pred_aux, aux_name=aux_name)
