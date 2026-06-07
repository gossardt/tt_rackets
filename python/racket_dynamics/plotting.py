from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import BounceDataset
from .models.base import ParameterPrediction
from .physics.bounce import est_alpha, get_beta, get_cor, get_kp, get_surface_v

METHOD_DISPLAY_NAMES = {
    "parametric_constant": "Const",
    "parametric_stateful": "Linear",
    "gp_parameters": "GP 1D",
    "gp_parameters_full": "GP",
    "mlp_parameters": "MLP 1D",
    "mlp_parameters_full": "MLP",
    "residual_mlp": "Residual 1D",
    "residual_mlp_full": "Residual",
}
PLOT_DPI = 300


def set_plt_params() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,
            "pgf.rcfonts": False,
            "font.size": 14,
            "figure.dpi": PLOT_DPI,
            "savefig.dpi": PLOT_DPI,
        }
    )


def display_method_name(method_name: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method_name, method_name.replace("_", " ").title())


def _aux_display_name(aux_name: str) -> str:
    if aux_name == "kp":
        return r"$k_p$"
    if aux_name == "alpha":
        return r"$\alpha$"
    return aux_name


def _measured_aux_parameter(dataset: BounceDataset, aux_name: str, cor_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if aux_name == "kp":
        kp_true = np.mean(np.stack(get_kp(dataset.v_in, dataset.w_in, dataset.v_out, dataset.w_out), axis=0), axis=0)
        x = get_surface_v(dataset.v_in, dataset.w_in)
        return kp_true, x, "$v_s$ [m/s]"
    if aux_name == "alpha":
        alpha_true = np.mean(
            np.stack(est_alpha(dataset.v_in, dataset.w_in, dataset.v_out, dataset.w_out), axis=0),
            axis=0,
        )
        x = get_beta(dataset.v_in, dataset.w_in, cor_true)
        return alpha_true, x, r"$\beta$"
    raise ValueError(f"Unsupported auxiliary parameter {aux_name}")


def aux_plot_filename(aux_name: str) -> str:
    return "kp_fit" if aux_name == "kp" else "alpha_fit"


def _aux_plot_color(dataset: BounceDataset, aux_name: str) -> tuple[np.ndarray, str]:
    if aux_name == "kp":
        return dataset.v_in[:, 2], "$v_z$ [m/s]"
    return get_surface_v(dataset.v_in, dataset.w_in), "$v_s$ [m/s]"


def _bin_to_grid(x_sorted: np.ndarray, y_sorted: np.ndarray, x_limits: tuple[float, float], num_bins: int = 80):
    x_min, x_max = x_limits
    edges = np.linspace(x_min, x_max, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned = np.full(num_bins, np.nan, dtype=float)
    for idx in range(num_bins):
        if idx == num_bins - 1:
            mask = (x_sorted >= edges[idx]) & (x_sorted <= edges[idx + 1])
        else:
            mask = (x_sorted >= edges[idx]) & (x_sorted < edges[idx + 1])
        if np.any(mask):
            binned[idx] = float(np.mean(y_sorted[mask]))
    return centers, binned


def _add_linear_fit_line(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    exclude_x_range: tuple[float, float] | None = None,
) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if exclude_x_range is not None:
        x_min_exclude, x_max_exclude = exclude_x_range
        mask &= ~((np.asarray(x) >= x_min_exclude) & (np.asarray(x) <= x_max_exclude))
    x_fit = np.asarray(x)[mask]
    y_fit = np.asarray(y)[mask]
    if x_fit.shape[0] < 2 or np.allclose(np.ptp(x_fit), 0.0):
        return

    slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
    x_line = np.linspace(float(np.min(x_fit)), float(np.max(x_fit)), 200)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="black",
        linewidth=1.5,
        label=f"Linear Fit: y = {slope:.4g}x + {intercept:.4g}",
    )


def save_cor_plot(dataset: BounceDataset, output_path: str | Path, title: str | None = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cor = get_cor(dataset.v_in, dataset.v_out)
    surface_v = get_surface_v(dataset.v_in, dataset.w_in)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(dataset.v_in[:, 2], cor, c=surface_v, cmap="coolwarm", alpha=0.7)
    _add_linear_fit_line(ax, dataset.v_in[:, 2], cor)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("$v_s$ [m/s]")
    ax.set_xlabel("$v_z$ [m/s]")
    ax.set_ylabel("COR")
    ax.set_xlim(-12.0, -3.0)
    ax.set_ylim(0.4, 0.9)
    ax.grid(True)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_observed_aux_plot(
    dataset: BounceDataset,
    aux_name: str,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    cor_true = get_cor(dataset.v_in, dataset.v_out)
    aux_true, aux_x, aux_xlabel = _measured_aux_parameter(dataset, aux_name, cor_true)
    aux_color_values, aux_color_label = _aux_plot_color(dataset, aux_name)
    _save_single_parameter_fit_plot(
        x=aux_x,
        measured_values=aux_true,
        color_values=aux_color_values,
        color_label=aux_color_label,
        method_predictions={},
        ylabel=_aux_display_name(aux_name),
        xlabel=aux_xlabel,
        measured_label=None,
        output_path=output_path,
        title=title,
        x_limits=(0.0, 15.0) if aux_name == "kp" else (0.0, 7.0),
        show_legend=True,
        show_linear_fit=True,
        linear_fit_exclude_x_range=(0.0, 0.5) if aux_name == "kp" else None,
    )


def save_observed_parameter_plots(
    dataset: BounceDataset,
    aux_name: str,
    cor_output_path: str | Path,
    aux_output_path: str | Path,
    cor_title: str | None = None,
    aux_title: str | None = None,
) -> None:
    save_cor_plot(dataset, cor_output_path, title=cor_title)
    save_observed_aux_plot(dataset, aux_name, aux_output_path, title=aux_title)


def save_error_diagnostics(
    dataset: BounceDataset,
    pred_v_out: np.ndarray,
    pred_w_out: np.ndarray,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    v_error = pred_v_out[:, [0, 2]] - dataset.v_out[:, [0, 2]]
    w_error = pred_w_out[:, 1] - dataset.w_out[:, 1]
    color = get_surface_v(dataset.v_in, dataset.w_in)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    if title:
        fig.suptitle(title)

    sc = axs[0, 0].scatter(dataset.w_in[:, 1], w_error, c=color)
    axs[0, 0].set_ylabel("$w_y'$ error [rad/s]")
    axs[0, 0].set_xlabel("$w_y$ [rad/s]")

    axs[0, 1].scatter(dataset.w_in[:, 1], v_error[:, 0], c=color)
    axs[0, 1].set_ylabel("$v_x'$ error [m/s]")
    axs[0, 1].set_xlabel("$w_y$ [rad/s]")

    axs[0, 2].scatter(dataset.w_in[:, 1], v_error[:, 1], c=color)
    axs[0, 2].set_ylabel("$v_z'$ error [m/s]")
    axs[0, 2].set_xlabel("$w_y$ [rad/s]")

    axs[1, 0].scatter(dataset.v_in[:, 0], w_error, c=color)
    axs[1, 0].set_ylabel("$w_y'$ error [rad/s]")
    axs[1, 0].set_xlabel("$v_x$ [m/s]")

    axs[1, 1].scatter(dataset.v_in[:, 0], v_error[:, 0], c=color)
    axs[1, 1].set_ylabel("$v_x'$ error [m/s]")
    axs[1, 1].set_xlabel("$v_x$ [m/s]")

    axs[1, 2].scatter(dataset.v_in[:, 0], v_error[:, 1], c=color)
    axs[1, 2].set_ylabel("$v_z'$ error [m/s]")
    axs[1, 2].set_xlabel("$v_x$ [m/s]")

    axs[2, 0].scatter(dataset.v_in[:, 2], w_error, c=color)
    axs[2, 0].set_ylabel("$w_y'$ error [rad/s]")
    axs[2, 0].set_xlabel("$v_z$ [m/s]")

    axs[2, 1].scatter(dataset.v_in[:, 2], v_error[:, 0], c=color)
    axs[2, 1].set_ylabel("$v_x'$ error [m/s]")
    axs[2, 1].set_xlabel("$v_z$ [m/s]")

    axs[2, 2].scatter(dataset.v_in[:, 2], v_error[:, 1], c=color)
    axs[2, 2].set_ylabel("$v_z'$ error [m/s]")
    axs[2, 2].set_xlabel("$v_z$ [m/s]")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.03, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="$v_s$ [m/s]")
    for i in range(3):
        for j in range(3):
            axs[i, j].grid(True)
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_method_comparison_plot(
    dataset: BounceDataset,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str | Path,
    title: str | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    x = dataset.v_in[:, 2]

    method_colors = plt.cm.tab10(np.linspace(0, 1, max(len(predictions), 1)))

    axs[0].scatter(x, dataset.v_out[:, 0], color="black", alpha=0.2, s=16, label="Data")
    axs[1].scatter(x, dataset.v_out[:, 2], color="black", alpha=0.2, s=16, label="Data")
    axs[2].scatter(x, dataset.w_out[:, 1], color="black", alpha=0.2, s=16, label="Data")

    for color, (method_name, (pred_v_out, pred_w_out)) in zip(method_colors, predictions.items()):
        label = display_method_name(method_name)
        axs[0].scatter(x, pred_v_out[:, 0], color=color, alpha=0.55, s=12, label=label)
        axs[1].scatter(x, pred_v_out[:, 2], color=color, alpha=0.55, s=12, label=label)
        axs[2].scatter(x, pred_w_out[:, 1], color=color, alpha=0.55, s=12, label=label)

    axs[0].set_ylabel("$v_x'$ [m/s]")
    axs[1].set_ylabel("$v_z'$ [m/s]")
    axs[2].set_ylabel("$w_y'$ [rad/s]")
    axs[2].set_xlabel("$v_z$ [m/s]")

    for ax in axs:
        ax.grid(True)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def _plot_parameter_prediction_lines(
    ax,
    x: np.ndarray,
    method_predictions: dict[str, tuple[np.ndarray, np.ndarray | None]],
    x_limits: tuple[float, float] | None = None,
) -> None:
    order = np.argsort(x)
    x_sorted = np.asarray(x)[order]
    x_min = float(np.min(x_sorted)) if x_limits is None else float(x_limits[0])
    x_max = float(np.max(x_sorted)) if x_limits is None else float(x_limits[1])
    grid_limits = (x_min, x_max)
    method_colors = plt.cm.tab10(np.linspace(0, 1, max(len(method_predictions), 1)))
    for color, (method_name, (predicted_values, predicted_std)) in zip(method_colors, method_predictions.items()):
        label = display_method_name(method_name)
        pred_sorted = np.asarray(predicted_values)[order]
        x_grid, pred_grid = _bin_to_grid(x_sorted, pred_sorted, grid_limits)
        valid = np.isfinite(pred_grid)
        ax.plot(x_grid[valid], pred_grid[valid], color=color, linewidth=1.5, label=label)

        if predicted_std is not None:
            std_sorted = np.asarray(predicted_std)[order]
            _, std_grid = _bin_to_grid(x_sorted, std_sorted, grid_limits)
            ci95 = 1.96 * std_grid
            ax.fill_between(
                x_grid[valid],
                (pred_grid - ci95)[valid],
                (pred_grid + ci95)[valid],
                color=color,
                alpha=0.15,
                label=f"{label} 95% CI",
            )


def _save_single_parameter_fit_plot(
    x: np.ndarray,
    measured_values: np.ndarray,
    color_values: np.ndarray,
    color_label: str,
    method_predictions: dict[str, tuple[np.ndarray, np.ndarray | None]],
    ylabel: str,
    xlabel: str,
    measured_label: str | None,
    output_path: str | Path,
    title: str | None = None,
    x_limits: tuple[float, float] | None = None,
    show_legend: bool = True,
    show_linear_fit: bool = False,
    linear_fit_exclude_x_range: tuple[float, float] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter_kwargs = {"c": color_values, "cmap": "coolwarm", "alpha": 0.5}
    if measured_label is not None:
        scatter_kwargs["label"] = measured_label
    scatter = ax.scatter(x, measured_values, **scatter_kwargs)
    _plot_parameter_prediction_lines(ax, x, method_predictions, x_limits=x_limits)
    if show_linear_fit:
        _add_linear_fit_line(ax, x, measured_values, exclude_x_range=linear_fit_exclude_x_range)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if ylabel == "COR":
        ax.set_ylim(0.4, 0.9)
    elif ylabel == r"$k_p$":
        ax.set_ylim(0.0, 0.003)
    elif ylabel == r"$\alpha$":
        ax.set_ylim(0.0, 0.8)
    ax.grid(True, linestyle=":", linewidth=0.7)
    if show_legend:
        ax.legend()

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)


def save_parameter_fit_plots(
    dataset: BounceDataset,
    parameter_predictions: ParameterPrediction,
    cor_output_path: str | Path,
    alpha_output_path: str | Path,
    cor_title: str | None = None,
    alpha_title: str | None = None,
    method_name: str = "estimate",
) -> None:
    save_parameter_fit_comparison_plots(
        dataset=dataset,
        parameter_predictions={method_name: parameter_predictions},
        cor_output_path=cor_output_path,
        alpha_output_path=alpha_output_path,
        cor_title=cor_title,
        alpha_title=alpha_title,
    )


def save_parameter_fit_comparison_plots(
    dataset: BounceDataset,
    parameter_predictions: dict[str, ParameterPrediction],
    cor_output_path: str | Path,
    alpha_output_path: str | Path,
    cor_title: str | None = None,
    alpha_title: str | None = None,
) -> None:
    cor_true = get_cor(dataset.v_in, dataset.v_out)
    surface_v = get_surface_v(dataset.v_in, dataset.w_in)
    cor_predictions = {
        method_name: (np.asarray(prediction.cor), prediction.cor_std)
        for method_name, prediction in parameter_predictions.items()
    }
    aux_name = next(iter(parameter_predictions.values())).aux_name
    aux_true, aux_x, aux_xlabel = _measured_aux_parameter(dataset, aux_name, cor_true)
    aux_color_values, aux_color_label = _aux_plot_color(dataset, aux_name)
    aux_predictions = {
        method_name: (np.asarray(prediction.aux), prediction.aux_std)
        for method_name, prediction in parameter_predictions.items()
    }
    _save_single_parameter_fit_plot(
        x=dataset.v_in[:, 2],
        measured_values=cor_true,
        color_values=surface_v,
        color_label="$v_s$ [m/s]",
        method_predictions=cor_predictions,
        ylabel="COR",
        xlabel="$v_z$ [m/s]",
        measured_label="Measurement",
        output_path=cor_output_path,
        title=cor_title,
        x_limits=(-12.0, -3.0),
    )
    _save_single_parameter_fit_plot(
        x=aux_x,
        measured_values=aux_true,
        color_values=aux_color_values,
        color_label=aux_color_label,
        method_predictions=aux_predictions,
        ylabel=_aux_display_name(aux_name),
        xlabel=aux_xlabel,
        measured_label=f"Measurement",
        output_path=alpha_output_path,
        title=alpha_title,
        x_limits=(0.0, 15.0) if aux_name == "kp" else (0.0, 7.0),
    )
