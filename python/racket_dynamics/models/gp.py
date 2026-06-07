"""
Utilities for GP.
"""

import joblib
import numpy as np
import torch
import gpytorch
from scipy.interpolate import RBFInterpolator

from .base import Estimator


def gp2fn(gp):
    x_dense = np.random.uniform(0, 10, size=(500, 2))
    y_pred = gp.predict(x_dense)
    return RBFInterpolator(x_dense, y_pred)


def save_fn(fn, path):
    joblib.dump(fn, path)


def read_fn(path):
    return joblib.load(path)


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(model, likelihood, train_x, train_y, steps=300, lr=0.05):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    best_state = None
    best_loss = float("inf")

    for _ in range(steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "likelihood": {
                    k: v.detach().clone() for k, v in likelihood.state_dict().items()
                },
            }

    # Refine with LBFGS to better maximize exact marginal likelihood.
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.3, max_iter=40)

    def closure():
        lbfgs.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        return loss

    lbfgs.step(closure)

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        refined_loss = float((-mll(model(train_x), train_y)).item())

    if best_state is not None and best_loss < refined_loss:
        model.load_state_dict(best_state["model"])
        likelihood.load_state_dict(best_state["likelihood"])


def eval_gp(model, x):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = model(x)
    return preds.mean, preds.variance.sqrt()


class GPScalarRegressor(Estimator):
    method_name = "gp_scalar"
    task_type = "parameter"

    def __init__(self):
        self.likelihood = None
        self.model = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, train_data):
        train_x, train_y = train_data
        self.x_mean = train_x.mean(dim=0, keepdim=True)
        self.x_std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.y_mean = train_y.mean()
        self.y_std = train_y.std().clamp_min(1e-6)

        train_x_norm = (train_x - self.x_mean) / self.x_std
        train_y_norm = (train_y - self.y_mean) / self.y_std
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPModel(train_x_norm, train_y_norm, self.likelihood)
        train_gp(self.model, self.likelihood, train_x_norm, train_y_norm)
        self.model.eval()
        self.likelihood.eval()
        return self

    def predict(self, x):
        x_norm = (x - self.x_mean) / self.x_std
        mean, _ = eval_gp(self.model, x_norm)
        return mean * self.y_std + self.y_mean

    def predict_uncertainty(self, x):
        x_norm = (x - self.x_mean) / self.x_std
        _, std = eval_gp(self.model, x_norm)
        return std * self.y_std
