from __future__ import annotations

from .baselines import GPParameterBaseline, ParametricConstantBaseline, ParametricStatefulBaseline
from .mlp import MLPParameterBaseline
from .residual import ResidualMLPBaseline


METHOD_NAMES = (
    "parametric_constant",
    "parametric_stateful",
    "gp_parameters",
    "gp_parameters_full",
    "mlp_parameters",
    "mlp_parameters_full",
    "residual_mlp",
    "residual_mlp_full",
)
DEFAULT_METHOD = "parametric_constant"


def create_method(method_name: str, racket_id: str):
    if method_name == "parametric_constant":
        return ParametricConstantBaseline(racket_id=racket_id)
    if method_name == "parametric_stateful":
        return ParametricStatefulBaseline(racket_id=racket_id)
    if method_name == "gp_parameters":
        return GPParameterBaseline(racket_id=racket_id, input_mode="single")
    if method_name == "gp_parameters_full":
        return GPParameterBaseline(racket_id=racket_id, input_mode="vs_vz")
    if method_name == "mlp_parameters":
        return MLPParameterBaseline(racket_id=racket_id, random_state=0, input_mode="vs_vz")
    if method_name == "mlp_parameters_full":
        return MLPParameterBaseline(racket_id=racket_id, random_state=0, input_mode="vs_vz")
    if method_name == "residual_mlp":
        return ResidualMLPBaseline(racket_id=racket_id, random_state=0, input_mode="vs_vz")
    if method_name == "residual_mlp_full":
        return ResidualMLPBaseline(racket_id=racket_id, random_state=0, input_mode="vs_vz")
    raise ValueError(f"Unknown method {method_name}")
