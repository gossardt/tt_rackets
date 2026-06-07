from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from ..configs.experiments import ExperimentConfig
from ..data import load_train_test_datasets
from ..models import ParameterEstimator, create_method
from ..plotting import (
    aux_plot_filename,
    display_method_name,
    save_cor_plot,
    save_error_diagnostics,
    save_parameter_fit_plots,
    set_plt_params,
)
from .metrics import compute_bounce_metrics
from .reports import ExperimentReport, ExperimentResult, RacketEvaluation, RacketReport

from ..models.registry import METHOD_NAMES

ALL_METHODS = METHOD_NAMES


def evaluate_method_on_racket(method_name: str, racket_id: str):
    train_dataset, test_dataset = load_train_test_datasets(racket_id)
    method = create_method(method_name, racket_id)
    method.fit(train_dataset)
    pred_v_out, pred_w_out = method.predict(test_dataset)
    metrics = compute_bounce_metrics(pred_v_out, pred_w_out, test_dataset.v_out, test_dataset.w_out)
    train_parameters = method.predict_parameters(train_dataset) if isinstance(method, ParameterEstimator) else None
    return train_dataset, test_dataset, pred_v_out, pred_w_out, metrics, train_parameters


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    racket_reports: list[RacketReport] = []
    evaluations: list[RacketEvaluation] = []
    if config.plot_dir is not None:
        set_plt_params()

    for racket_id in config.rackets:
        method_label = display_method_name(config.method.name)
        train_dataset, test_dataset, pred_v_out, pred_w_out, metrics, train_parameters = evaluate_method_on_racket(
            config.method.name,
            racket_id,
        )
        if config.plot_dir is not None:
            plot_root = Path(config.plot_dir) / config.method.name
            save_cor_plot(
                test_dataset,
                plot_root / f"racket_{racket_id}_cor.png",
                title=f"Racket {racket_id} COR",
            )
            if train_parameters is not None:
                aux_label = "k_p" if train_parameters.aux_name == "kp" else "alpha"
                aux_filename = aux_plot_filename(train_parameters.aux_name)
                save_parameter_fit_plots(
                    train_dataset,
                    train_parameters,
                    plot_root / f"racket_{racket_id}_cor_fit.png",
                    plot_root / f"racket_{racket_id}_{aux_filename}.png",
                    cor_title=f"{method_label} racket {racket_id} training COR",
                    alpha_title=f"{method_label} racket {racket_id} training {aux_label}",
                    method_name=config.method.name,
                )
            save_error_diagnostics(
                test_dataset,
                pred_v_out,
                pred_w_out,
                plot_root / f"racket_{racket_id}_errors.png",
                title=f"{method_label} racket {racket_id}",
            )
        racket_report = RacketReport(
            racket_id=racket_id,
            num_train=len(train_dataset),
            num_test=len(test_dataset),
            metrics=metrics,
        )
        racket_reports.append(racket_report)
        evaluations.append(
            RacketEvaluation(
                racket_id=racket_id,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                pred_v_out=pred_v_out,
                pred_w_out=pred_w_out,
                train_parameters=train_parameters,
                report=racket_report,
            )
        )

    velocity_means = np.array([report.metrics.velocity_mae_mean for report in racket_reports], dtype=float)
    spin_means = np.array([report.metrics.spin_mae_mean for report in racket_reports], dtype=float)

    report = ExperimentReport(
        method_name=config.method.name,
        split_name=config.split_name,
        rackets=racket_reports,
        mean_velocity_mae=float(np.mean(velocity_means)),
        std_velocity_mae=float(np.std(velocity_means)),
        mean_spin_mae=float(np.mean(spin_means)),
        std_spin_mae=float(np.std(spin_means)),
    )

    if config.output_path is not None:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2))

    return ExperimentResult(report=report, evaluations=evaluations)
