from __future__ import annotations

from .base import DirectBounceEstimator, Estimator, ParameterEstimator, ParameterPrediction

__all__ = [
    "BounceMethod",
    "create_method",
    "DEFAULT_METHOD",
    "DirectBounceEstimator",
    "Estimator",
    "GPParameterBaseline",
    "GPModel",
    "GPScalarRegressor",
    "MLPParameterBaseline",
    "METHOD_NAMES",
    "ParameterEstimator",
    "ParameterPrediction",
    "ParametricConstantBaseline",
    "ParametricStatefulBaseline",
    "ResidualMLPBaseline",
    "eval_gp",
    "train_gp",
]


def __getattr__(name: str):
    if name in {"BounceMethod", "GPParameterBaseline", "ParametricConstantBaseline", "ParametricStatefulBaseline"}:
        from .baselines import BounceMethod, GPParameterBaseline, ParametricConstantBaseline, ParametricStatefulBaseline

        exports = {
            "BounceMethod": BounceMethod,
            "GPParameterBaseline": GPParameterBaseline,
            "ParametricConstantBaseline": ParametricConstantBaseline,
            "ParametricStatefulBaseline": ParametricStatefulBaseline,
        }
        return exports[name]
    if name in {"GPModel", "GPScalarRegressor", "eval_gp", "train_gp"}:
        from .gp import GPModel, GPScalarRegressor, eval_gp, train_gp

        exports = {
            "GPModel": GPModel,
            "GPScalarRegressor": GPScalarRegressor,
            "eval_gp": eval_gp,
            "train_gp": train_gp,
        }
        return exports[name]
    if name == "MLPParameterBaseline":
        from .mlp import MLPParameterBaseline

        return MLPParameterBaseline
    if name in {"DEFAULT_METHOD", "METHOD_NAMES", "create_method"}:
        from .registry import DEFAULT_METHOD, METHOD_NAMES, create_method

        exports = {
            "DEFAULT_METHOD": DEFAULT_METHOD,
            "METHOD_NAMES": METHOD_NAMES,
            "create_method": create_method,
        }
        return exports[name]
    if name == "ResidualMLPBaseline":
        from .residual import ResidualMLPBaseline

        return ResidualMLPBaseline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
