"""
Builds the different racket models.
"""

from ..models.base import DirectBounceEstimator
from ..models.parametric import piecewise_lin_fn
from .bounce import elastic_bounce, get_surface_v, mixed_coulomb_bounce


class Racket(DirectBounceEstimator):
    method_name = "parametric_racket"
    task_type = "direct"

    def __init__(self, racket_id, cor_coeffs, friction_model, friction_coeffs) -> None:
        self.id = racket_id
        self.cor_coeffs = cor_coeffs
        self.friction = friction_model
        self.friction_coeffs = friction_coeffs

    def fit(self, train_data):
        return self

    def get_cor(self, v_in, w_in):
        s = get_surface_v(v_in, w_in)
        return self.cor_coeffs[0] * s + self.cor_coeffs[1] * v_in[:, 2] + self.cor_coeffs[2]

    def get_kp(self, v_in, w_in):
        s = get_surface_v(v_in, w_in)
        return self.friction_coeffs[0] * s + self.friction_coeffs[1] * v_in[:, 2] + self.friction_coeffs[2]

    def get_kp_piece(self, v_in, w_in):
        s = get_surface_v(v_in, w_in)
        return piecewise_lin_fn(s, *self.friction_coeffs)

    def predict(self, v_in, w_in):
        cor = self.get_cor(v_in, w_in)
        if self.friction == "coulomb":
            mu = self.friction_coeffs[0]
            return mixed_coulomb_bounce(
                v_in,
                w_in,
                cor,
                mu,
                mean_dist=self.friction_coeffs[1],
                std_dist=self.friction_coeffs[2],
            )
        if self.friction == "elastic":
            kp = self.get_kp(v_in, w_in)
            return elastic_bounce(v_in, w_in, cor, kp)
        if self.friction == "elastic_piece":
            kp = self.get_kp_piece(v_in, w_in)
            return elastic_bounce(v_in, w_in, cor, kp)
        raise ValueError(f"Friction model {self.friction} not implemented")

    def bounce(self, v_in, w_in):
        return self.predict(v_in, w_in)

