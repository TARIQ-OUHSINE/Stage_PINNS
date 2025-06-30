import torch
from torch import nn
from torch.nn import MSELoss
import torch.optim as optim

import numpy as np

from polpinn.simple_model.physics import FickResidual

mse = MSELoss()


class PDELoss:
    def __init__(self, P, D, T, P0, radius, t_max, n_samples):
        self.fick_residual = FickResidual(P=P, D=D, T=T, P0=P0)
        self.n_samples = n_samples
        self.radius = radius
        self.t_max = t_max

    def __call__(self):
        # X = torch.rand((self.n_samples, 2), requires_grad=True)
        # X = X * torch.tensor([self.radius, self.t_max])
        X = torch.cat(
            [
                torch.arange(1, self.n_samples + 1).view(-1, 1)
                / self.n_samples
                * self.radius,
                torch.arange(1, self.n_samples + 1).view(-1, 1)
                / self.n_samples
                * self.t_max,
                # torch.arange(self.n_samples).view(-1, 1)
                # / (self.n_samples - 1)
                # * self.t_max,
            ],
            dim=1,
        )
        X.requires_grad_(True)
        res = self.fick_residual(X=X)
        return mse(res, torch.zeros_like(res))


class AverageInLoss:
    def __init__(self, G, times, values, radius):
        self.times = times.view(-1, 1)
        self.values = values.view(-1, 1)
        self.radius = radius
        self.G = G

    def __call__(self):
        X = torch.cat(
            [torch.full_like(self.times, fill_value=self.radius), self.times], dim=1
        )
        X.requires_grad_(True)
        return mse(self.G(X), self.values)


class AverageOutLoss:
    def __init__(self, P, times, values, radius):
        self.times = times.view(-1, 1)
        self.values = values.view(-1, 1)
        self.radius = radius
        self.P = P

    def __call__(self):
        X = torch.cat(
            [torch.full_like(self.times, fill_value=self.radius), self.times],
            dim=1,
        )
        X.requires_grad_(True)
        p_tensor = self.P(X)
        return mse(p_tensor, self.values)


class InitLoss:
    def __init__(self, P, n_samples, radius):
        self.n_samples = n_samples
        self.radius = radius
        self.P = P

    def __call__(self):
        # r_tensor = torch.rand(self.n_samples, requires_grad=True) * self.radius
        r_tensor = torch.linspace(0, self.radius, self.n_samples, requires_grad=True)
        r_tensor = r_tensor.view(-1, 1)
        X = torch.cat([r_tensor, torch.zeros_like(r_tensor, requires_grad=True)], dim=1)
        p_tensor = self.P(X)
        # res = G(X)
        return mse(p_tensor, torch.zeros_like(p_tensor))


class BoundaryLoss:
    def __init__(self, P, n_samples, t_max):
        self.n_samples = n_samples
        self.t_max = t_max
        self.P = P

    def __call__(self):
        # t_tensor = torch.rand(self.n_samples, requires_grad=True) * self.t_max
        t_tensor = torch.linspace(0, self.t_max, self.n_samples, requires_grad=True)
        t_tensor = t_tensor.view(-1, 1)
        X0 = torch.cat([torch.zeros_like(t_tensor), t_tensor], dim=1)
        X = X0

        p_tensor = self.P(X)
        dP = torch.autograd.grad(
            p_tensor, X, grad_outputs=torch.ones_like(p_tensor), create_graph=True
        )[0]
        dP_dr = dP[:, 0].view(-1, 1)

        return mse(dP_dr, torch.zeros_like(dP_dr))


def compute_loss(
    model_G,
    model_P,
    data_in,
    data_out,
    R,
    t_max,
    T1,
    P0,
    D_f,
    n_samples,
    extra_radius_ratio,
):
    losses = {}
    losses["average_in"] = AverageInLoss(G=model_G, **data_in, radius=R)()
    # L_bord = AverageOutLoss(**data_out, radius=R + 0.009)(model_P)
    # L_ini = InitLoss(n_samples=n_samples, radius=R + 0.009)(model_P)
    losses["average_out"] = AverageOutLoss(
        P=model_P, **data_out, radius=R * extra_radius_ratio
    )()
    losses["init"] = InitLoss(
        P=model_P, n_samples=n_samples, radius=R * extra_radius_ratio
    )()
    # L_bord = AverageOutLoss(**data_out, radius=R + 0.009e-7)(model_P)
    # L_ini = InitLoss(n_samples=n_samples, radius=R + 0.009e-7)(model_P)
    losses["pde"] = PDELoss(
        P=model_P,
        D=D_f,
        T=T1,
        P0=P0,
        radius=R,
        t_max=t_max,
        n_samples=n_samples,
    )()
    losses["bnd"] = BoundaryLoss(
        P=model_P,
        t_max=t_max,
        n_samples=n_samples,
    )()
    losses["total"] = sum(losses.values())
    return losses
