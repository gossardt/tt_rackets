from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BounceMetrics:
    velocity_mae_mean: float
    velocity_mae_std: float
    spin_mae_mean: float
    spin_mae_std: float


def compute_bounce_metrics(pred_v_out, pred_w_out, v_out, w_out) -> BounceMetrics:
    v_error = pred_v_out[:, [0, 2]] - v_out[:, [0, 2]]
    w_error = pred_w_out[:, 1] - w_out[:, 1]
    velocity_mae = np.linalg.norm(v_error, axis=1)
    spin_mae = np.abs(w_error)
    return BounceMetrics(
        velocity_mae_mean=float(np.mean(velocity_mae)),
        velocity_mae_std=float(np.std(velocity_mae)),
        spin_mae_mean=float(np.mean(spin_mae)),
        spin_mae_std=float(np.std(spin_mae)),
    )

