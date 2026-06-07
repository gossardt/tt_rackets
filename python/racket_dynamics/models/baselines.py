from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data import BounceDataset
from ..configs import RACKET_PARAMS
from ..physics.bounce import (
    coulomb_bounce,
    est_alpha,
    get_alpha,
    get_beta,
    get_cor,
    get_kp,
    get_mixed_alpha,
    get_surface_v,
    lin_bounce,
)
from ..physics.constants import M
from ..physics.racket import Racket
from .base import ParameterEstimator, ParameterPrediction
from .gp import GPScalarRegressor


class BounceMethod:
    method_name = "bounce_method"
    deterministic = True

    def fit(self, dataset: BounceDataset):
        return self

    def predict(self, dataset: BounceDataset):
        raise NotImplementedError

    def _derived_parameter_prediction(self, dataset: BounceDataset) -> ParameterPrediction:
        pred_v_out, pred_w_out = self.predict(dataset)
        pred_cor = get_cor(dataset.v_in, pred_v_out)
        if getattr(self, "racket_id", None) == "10":
            pred_aux = np.mean(
                np.stack(est_alpha(dataset.v_in, dataset.w_in, pred_v_out, pred_w_out), axis=0),
                axis=0,
            )
            return ParameterPrediction(cor=pred_cor, aux=pred_aux, aux_name="alpha")

        pred_aux = np.mean(
            np.stack(get_kp(dataset.v_in, dataset.w_in, pred_v_out, pred_w_out), axis=0),
            axis=0,
        )
        return ParameterPrediction(cor=pred_cor, aux=pred_aux, aux_name="kp")


class DerivedParameterEstimator(BounceMethod, ParameterEstimator):
    def predict_parameters(self, dataset: BounceDataset) -> ParameterPrediction:
        return self._derived_parameter_prediction(dataset)


@dataclass
class ParametricConstantBaseline(DerivedParameterEstimator):
    racket_id: str
    method_name: str = "parametric_constant"

    def __post_init__(self):
        params = RACKET_PARAMS[self.racket_id]
        self.model = Racket(
            int(self.racket_id),
            params["mean_cor"],
            params["friction_model"],
            params["mean_fr"],
        )

    def predict(self, dataset: BounceDataset):
        if self.model.friction == "coulomb":
            cor = self.model.get_cor(dataset.v_in, dataset.w_in)
            mu = self.model.friction_coeffs[0]
            return coulomb_bounce(dataset.v_in, dataset.w_in, cor, mu)
        return self.model.bounce(dataset.v_in, dataset.w_in)

    def predict_parameters(self, dataset: BounceDataset) -> ParameterPrediction:
        cor = self.model.get_cor(dataset.v_in, dataset.w_in)
        if self.model.friction == "elastic":
            aux = self.model.get_kp(dataset.v_in, dataset.w_in)
            aux_name = "kp"
        elif self.model.friction == "elastic_piece":
            aux = self.model.get_kp_piece(dataset.v_in, dataset.w_in)
            aux_name = "kp"
        elif self.model.friction == "coulomb":
            aux = np.minimum(get_alpha(dataset.v_in, dataset.w_in, cor, self.model.friction_coeffs[0]), 0.4)
            aux_name = "alpha"
        else:
            raise ValueError(f"Unsupported friction model {self.model.friction}")
        return ParameterPrediction(cor=cor, aux=aux, aux_name=aux_name)


@dataclass
class ParametricStatefulBaseline(DerivedParameterEstimator):
    racket_id: str
    method_name: str = "parametric_stateful"

    def __post_init__(self):
        params = RACKET_PARAMS[self.racket_id]
        self.model = Racket(
            int(self.racket_id),
            params["cor_coeffs"],
            params["friction_model"],
            params["friction_coeffs"],
        )

    def predict(self, dataset: BounceDataset):
        return self.model.bounce(dataset.v_in, dataset.w_in)

    def predict_parameters(self, dataset: BounceDataset) -> ParameterPrediction:
        cor = self.model.get_cor(dataset.v_in, dataset.w_in)
        if self.model.friction == "elastic":
            aux = self.model.get_kp(dataset.v_in, dataset.w_in)
            aux_name = "kp"
        elif self.model.friction == "elastic_piece":
            aux = self.model.get_kp_piece(dataset.v_in, dataset.w_in)
            aux_name = "kp"
        elif self.model.friction == "coulomb":
            beta = get_beta(dataset.v_in, dataset.w_in, cor)
            aux = get_mixed_alpha(
                beta,
                self.model.friction_coeffs[0],
                self.model.friction_coeffs[1],
                self.model.friction_coeffs[2],
            )
            aux_name = "alpha"
        else:
            raise ValueError(f"Unsupported friction model {self.model.friction}")
        return ParameterPrediction(cor=cor, aux=aux, aux_name=aux_name)


class GPParameterBaseline(BounceMethod, ParameterEstimator):
    method_name = "gp_parameters"
    deterministic = False

    def __init__(self, racket_id: str, input_mode: str = "single"):
        self.racket_id = racket_id
        self.friction_model = RACKET_PARAMS[racket_id]["friction_model"]
        self.input_mode = input_mode
        self.cor_gp = GPScalarRegressor()
        self.aux_gp = GPScalarRegressor()

    def _make_cor_features(self, dataset: BounceDataset) -> np.ndarray:
        if self.input_mode == "single":
            return dataset.v_in[:, 2:3]
        if self.input_mode == "vs_vz":
            s = get_surface_v(dataset.v_in, dataset.w_in)
            return np.column_stack([dataset.v_in[:, 2], s])
        raise ValueError(f"Unsupported input mode {self.input_mode}")

    def _make_aux_features(self, dataset: BounceDataset) -> np.ndarray:
        s = get_surface_v(dataset.v_in, dataset.w_in)
        if self.input_mode == "single":
            return s.reshape(-1, 1)
        if self.input_mode == "vs_vz":
            return np.column_stack([dataset.v_in[:, 2], s])
        raise ValueError(f"Unsupported input mode {self.input_mode}")

    def fit(self, dataset: BounceDataset):
        cor = get_cor(dataset.v_in, dataset.v_out)
        cor_x = torch.tensor(self._make_cor_features(dataset), dtype=torch.float)
        cor_y = torch.tensor(cor, dtype=torch.float)
        self.cor_gp.fit((cor_x, cor_y))

        aux_x = torch.tensor(self._make_aux_features(dataset), dtype=torch.float)
        if self.friction_model == "coulomb":
            aux_targets = np.mean(
                np.stack(est_alpha(dataset.v_in, dataset.w_in, dataset.v_out, dataset.w_out), axis=0),
                axis=0,
            )
            aux_y = torch.tensor(aux_targets, dtype=torch.float)
            self.aux_gp.fit((aux_x, aux_y))
        else:
            aux_targets = np.mean(
                np.stack(get_kp(dataset.v_in, dataset.w_in, dataset.v_out, dataset.w_out), axis=0),
                axis=0,
            )
            aux_y = torch.tensor(aux_targets, dtype=torch.float)
            self.aux_gp.fit((aux_x, aux_y))
        return self

    def predict(self, dataset: BounceDataset):
        cor_x = torch.tensor(self._make_cor_features(dataset), dtype=torch.float)
        aux_x = torch.tensor(self._make_aux_features(dataset), dtype=torch.float)
        pred_cor = self.cor_gp.predict(cor_x).numpy()
        pred_aux = self.aux_gp.predict(aux_x).numpy()
        pred_alpha = pred_aux if self.friction_model == "coulomb" else pred_aux / M
        return lin_bounce(dataset.v_in, dataset.w_in, pred_cor, pred_alpha)

    def predict_parameters(self, dataset: BounceDataset):
        cor_x_np = self._make_cor_features(dataset).astype(float)
        aux_x_np = self._make_aux_features(dataset).astype(float)
        # Avoid querying the exact stored training inputs when producing fit plots.
        cor_x_np[:, 0] += np.linspace(-1e-6, 1e-6, cor_x_np.shape[0])
        aux_x_np[:, 0] += np.linspace(-1e-6, 1e-6, aux_x_np.shape[0])
        cor_x = torch.tensor(cor_x_np, dtype=torch.float)
        aux_x = torch.tensor(aux_x_np, dtype=torch.float)
        aux_name = "alpha" if self.friction_model == "coulomb" else "kp"
        return ParameterPrediction(
            cor=self.cor_gp.predict(cor_x).numpy(),
            aux=self.aux_gp.predict(aux_x).numpy(),
            cor_std=self.cor_gp.predict_uncertainty(cor_x).numpy(),
            aux_std=self.aux_gp.predict_uncertainty(aux_x).numpy(),
            aux_name=aux_name,
        )
