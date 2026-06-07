from __future__ import annotations

from dataclasses import dataclass

from .rackets import DEFAULT_RACKETS
@dataclass(frozen=True)
class MethodConfig:
    name: str


@dataclass(frozen=True)
class ExperimentConfig:
    method: MethodConfig
    rackets: tuple[str, ...] = tuple(DEFAULT_RACKETS)
    split_name: str = "precomputed"
    output_path: str | None = None
    plot_dir: str | None = None


DEFAULT_SPLIT = "precomputed"


def make_experiment_config(
    method_name: str,
    rackets: tuple[str, ...],
    split_name: str = "precomputed",
    output_path: str | None = None,
    plot_dir: str | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        method=MethodConfig(name=method_name),
        rackets=rackets,
        split_name=split_name,
        output_path=output_path,
        plot_dir=plot_dir,
    )
