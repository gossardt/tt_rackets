"""
Racket models.
"""

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from .constants import I, M, R


def get_cor(v_in, v_out):
    return -v_out[:, 2] / v_in[:, 2]


def fit_cor(v_in, cor):
    model = LinearRegression()
    model.fit(v_in[:, 2].reshape(-1, 1), cor)
    return model.coef_[0], model.intercept_


def get_kp(v_in, w_in, v_out, w_out):
    kp_v = M * (v_out[:, 0] - v_in[:, 0]) / ((R * w_in[:, 1]) - (v_in[:, 0]))
    kp_w = I * (w_out[:, 1] - w_in[:, 1]) / ((R * v_in[:, 0]) - (R**2 * w_in[:, 1]))
    return kp_v, kp_w


def lin_bounce(v_in, w_in, cor, alpha):
    v_out = np.zeros_like(v_in)
    w_out = np.zeros_like(w_in)
    for i, (a, c) in enumerate(zip(alpha, cor)):
        A = np.array([[1 - a, 0, 0], [0, 1 - a, 0], [0, 0, -c]])
        B = np.array([[0, a * R, 0], [-a * R, 0, 0], [0, 0, 0]])
        C = np.array([[0, -3 / 2 * a / R, 0], [3 / 2 * a / R, 0, 0], [0, 0, 0]])
        D = np.array([[1 - 3 / 2 * a, 0, 0], [0, 1 - 3 / 2 * a, 0], [0, 0, 1]])
        v_out[i] = A @ v_in[i] + B @ w_in[i]
        w_out[i] = C @ v_in[i] + D @ w_in[i]
    return v_out, w_out


def elastic_bounce(v_in, w_in, cor, friction_coef):
    kp = friction_coef
    kv = kp / M
    kw = kp / I
    n = v_in.shape[0]

    v_out = np.zeros_like(v_in)
    w_out = np.zeros_like(w_in)

    if isinstance(cor, np.ndarray):
        for i in range(n):
            Bv = np.array([[0, kv[i] * R, 0], [-kv[i] * R, 0, 0], [0, 0, 0]])
            Aw = np.array([[0, -kw[i] * R, 0], [kw[i] * R, 0, 0], [0, 0, 0]])
            Bw = np.array(
                [[1 - kw[i] * R**2, 0, 0], [0, 1 - kw[i] * R**2, 0], [0, 0, 1]]
            )
            Av = np.array([[1 - kv[i], 0, 0], [0, 1 - kv[i], 0], [0, 0, -cor[i]]])
            v_out[i] = Av @ v_in[i] + Bv @ w_in[i]
            w_out[i] = Aw @ v_in[i] + Bw @ w_in[i]
    else:
        Bv = np.array([[0, kv * R, 0], [-kv * R, 0, 0], [0, 0, 0]])
        Aw = np.array([[0, -kw * R, 0], [kw * R, 0, 0], [0, 0, 0]])
        Bw = np.array([[1 - kw * R**2, 0, 0], [0, 1 - kw * R**2, 0], [0, 0, 1]])
        Av = np.array([[1 - kv, 0, 0], [0, 1 - kv, 0], [0, 0, -cor]])
        v_out = (Av @ v_in.T + Bv @ w_in.T).T
        w_out = (Aw @ v_in.T + Bw @ w_in.T).T

    return v_out, w_out


def get_surface_v(v_in, w_in):
    return np.sqrt((v_in[:, 0] - R * w_in[:, 1]) ** 2 + (v_in[:, 1] + R * w_in[:, 0]) ** 2)


def get_alpha(v_in, w_in, cor, mu):
    surf_v = get_surface_v(v_in, w_in)
    alpha = mu * (1 + cor) * np.abs(v_in[:, 2]) / surf_v
    return alpha


def get_beta(v_in, w_in, cor):
    surf_v = get_surface_v(v_in, w_in)
    beta = (1 + cor) * np.abs(v_in[:, 2]) / surf_v
    return beta


def est_alpha(v_in, w_in, v_out, w_out):
    alpha1 = (v_out[:, 0] - v_in[:, 0]) / (R * w_in[:, 1] - v_in[:, 0])
    alpha2 = 2 / 3 * (w_out[:, 1] - w_in[:, 1]) / (v_in[:, 0] / R - w_in[:, 1])
    return alpha1, alpha2


def get_mixed_alpha(beta, mu, mean_dist, std_dist):
    norm_dist = norm(loc=mean_dist, scale=std_dist)
    k = norm_dist.cdf(beta)
    alpha = ((1 - k) * mu * beta) + (k * 0.4)
    return alpha


def get_mu(v_in, w_in, v_out, w_out, cor):
    s = get_surface_v(v_in, w_in)
    mu1 = (
        (v_in[:, 0] - v_out[:, 0])
        * s
        / ((1 + cor) * v_in[:, 2] * (v_in[:, 0] - R * w_in[:, 1]))
    )
    mu2 = (
        (w_in[:, 1] - w_out[:, 1])
        * 2
        * R
        * s
        / (3 * (1 + cor) * v_in[:, 2] * (-v_in[:, 0] + R * w_in[:, 1]))
    )
    return mu1, mu2


def coulomb_bounce(v_in, w_in, cor, mu):
    v_out = np.zeros_like(v_in)
    w_out = np.zeros_like(w_in)
    alpha = get_alpha(v_in, w_in, cor, mu)

    for i, a in enumerate(alpha):
        c = cor[i] if isinstance(cor, np.ndarray) else cor
        if a > 0.4:
            A = np.array([[0.6, 0, 0], [0, 0.6, 0], [0, 0, -c]])
            B = np.array([[0, 0.4 * R, 0], [-0.4 * R, 0, 0], [0, 0, 0]])
            C = np.array([[0, -0.6 / R, 0], [0.6 / R, 0, 0], [0, 0, 0]])
            D = np.array([[0.4, 0, 0], [0, 0.4, 0], [0, 0, 1]])
        else:
            A = np.array([[1 - a, 0, 0], [0, 1 - a, 0], [0, 0, -c]])
            B = np.array([[0, a * R, 0], [-a * R, 0, 0], [0, 0, 0]])
            C = np.array([[0, -3 / 2 * a / R, 0], [3 / 2 * a / R, 0, 0], [0, 0, 0]])
            D = np.array([[1 - 3 / 2 * a, 0, 0], [0, 1 - 3 / 2 * a, 0], [0, 0, 1]])
        v_out[i] = A @ v_in[i] + B @ w_in[i]
        w_out[i] = C @ v_in[i] + D @ w_in[i]

    return v_out, w_out


def mixed_coulomb_bounce(v_in, w_in, cor, mu, mean_dist=1, std_dist=0.5):
    v_out = np.zeros_like(v_in)
    w_out = np.zeros_like(w_in)
    beta = get_beta(v_in, w_in, cor)
    alpha = get_mixed_alpha(beta, mu, mean_dist, std_dist)

    for i, a in enumerate(alpha):
        c = cor[i] if isinstance(cor, np.ndarray) else cor
        A = np.array([[1 - a, 0, 0], [0, 1 - a, 0], [0, 0, -c]])
        B = np.array([[0, a * R, 0], [-a * R, 0, 0], [0, 0, 0]])
        C = np.array([[0, -3 / 2 * a / R, 0], [3 / 2 * a / R, 0, 0], [0, 0, 0]])
        D = np.array([[1 - 3 / 2 * a, 0, 0], [0, 1 - 3 / 2 * a, 0], [0, 0, 1]])
        v_out[i] = A @ v_in[i] + B @ w_in[i]
        w_out[i] = C @ v_in[i] + D @ w_in[i]

    return v_out, w_out

