import torch
from torch import nn
from torch.nn import MSELoss
import torch.optim as optim

import numpy as np

from polpinn.simple_model.physics import FickResidual
from polpinn.simple_model.model import P_from_G

mse = MSELoss()


class PDELoss:
    def __init__(self, D, T, P0, system_radius, t_max, n_samples):
        self.fick_residual = FickResidual(D=D, T=T, P0=P0)
        self.n_samples = n_samples
        self.system_radius = system_radius
        self.t_max = t_max

    def __call__(self, G, R):
        X = torch.rand((self.n_samples, 2), requires_grad=True)
        X = X * torch.tensor([self.system_radius, self.t_max])
        p = P_from_G(G(X), X)
        res = self.fick_residual(P=p, X=X)
        return mse(res, torch.zeros_like(res))


class AverageInLoss:
    def __init__(self, times, values, radius):
        self.times = times.view(-1, 1)
        self.values = values.view(-1, 1)
        self.radius = radius

    def __call__(self, G):
        X = torch.cat(
            [torch.full_like(self.times, fill_value=self.radius), self.times], dim=1
        )
        return mse(G(X), self.values)


class AverageOutLoss:
    def __init__(self, in_data, out_data, radius, system_radius):
        times, in_ind, out_ind = np.intersect1d(
            in_data["times"], out_data["times"], return_indices=True
        )
        self.times = torch.tensor(times).view(-1, 1)
        in_values = in_data["values"][in_ind].view(-1, 1)
        out_values = out_data["values"][out_ind].view(-1, 1)
        self.values = (
            (system_radius**3 - radius**3) / system_radius**3 * out_values
            + radius**3 / system_radius**3 * in_values
        )
        self.system_radius = system_radius

    def __call__(self, G):
        X = torch.cat(
            [torch.full_like(self.times, fill_value=self.system_radius), self.times],
            dim=1,
        )
        return mse(G(X), self.values)


class InitLoss:
    def __init__(self, n_samples, radius, P0):
        self.n_samples = n_samples
        self.radius = radius
        self.P0 = P0

    def __call__(self, G):
        r_tensor = torch.rand(self.n_samples, requires_grad=True) * self.radius
        r_tensor = r_tensor.view(-1, 1)
        X = torch.cat([r_tensor, torch.zeros_like(r_tensor)], dim=1)
        p = P_from_G(G(X), X)
        # res = G(X)
        return mse(p, torch.full_like(p, fill_value=self.P0))


class BoundaryLoss:
    def __init__(self, n_samples, system_radius, t_max):
        self.n_samples = n_samples
        self.system_radius = system_radius
        self.t_max = t_max

    def __call__(self, G):
        t_tensor = torch.rand(self.n_samples, requires_grad=True) * self.t_max
        t_tensor = t_tensor.view(-1, 1)
        X0 = torch.cat([torch.zeros_like(t_tensor), t_tensor], dim=1)
        XR = torch.cat(
            [torch.full_like(t_tensor, fill_value=self.system_radius), t_tensor], dim=1
        )
        X = torch.cat([X0, XR], dim=0)

        p = P_from_G(G(X), X)
        dP = torch.autograd.grad(
            p, X, grad_outputs=torch.ones_like(p), create_graph=True
        )[0]
        dP_dr = dP[:, 0].view(-1, 1)

        return mse(dP_dr, torch.zeros_like(dP_dr))


if __name__ == "__main__":
    from tqdm import tqdm

    import matplotlib.pyplot as plt

    from model import G_MLP
    from data import load_data, data_augmentation

    data = load_data()

    params_pinns = {
        "n_layers": 2,
        "n_neurons_per_layer": 32,
        "lr": 0.001,
        "lr_R": 0.0001,
        # "epoch": 20000,
        "epoch": 100,
        "var_R": True,
    }

    # MODIF
    data["params"]["D_j"] = 2.5807e-16
    data["params"]["T_B"] = 0.1
    data["params"]["R_bis"] = 225e-9
    data["params"]["R_bis"] = data["params"]["rayon_initialisation"]
    data["params"]["rayon_initialisation"] = 100e-9

    R = data["params"]["rayon_initialisation"]
    D = data["params"]["D_f"]
    T = data["params"]["T_1"]
    P0 = data["params"]["P0_f"]
    system_radius = data["params"]["R_bis"]
    n_samples = 1000
    t_max = data["params"]["def_t"]

    # Data augmentation
    data_aug = {}
    data_aug["average_in"] = data_augmentation(mono=False)
    data_aug["average_in"].fit(**data["average_in"])

    data_aug["average_out"] = data_augmentation(mono=False)
    data_aug["average_out"].fit(**data["average_out"])

    t_data = torch.rand(n_samples) * t_max
    augmentated_data = {
        k: {"times": t_data, "values": data_aug[k](t_data)}
        for k in ("average_in", "average_out")
    }

    fig, axes = plt.subplots(1, 2)
    axes[0].scatter(
        data["average_in"]["times"], data["average_in"]["values"], label="data in"
    )
    axes[0].scatter(
        augmentated_data["average_in"]["times"],
        augmentated_data["average_in"]["values"],
        label="aug data in",
    )
    axes[0].legend()
    axes[1].scatter(
        data["average_out"]["times"], data["average_out"]["values"], label="data out"
    )
    axes[1].scatter(
        augmentated_data["average_out"]["times"],
        augmentated_data["average_out"]["values"],
        label="aug data out",
    )
    axes[1].legend()

    ##
    pde_loss = PDELoss(
        D=D, T=T, P0=P0, system_radius=system_radius, t_max=t_max, n_samples=n_samples
    )
    average_in_loss = AverageInLoss(**augmentated_data["average_in"], radius=R)
    average_out_loss = AverageOutLoss(
        in_data=augmentated_data["average_in"],
        out_data=augmentated_data["average_out"],
        radius=R,
        system_radius=system_radius,
    )
    init_loss = InitLoss(n_samples=n_samples, radius=R, P0=P0)
    boundary_loss = BoundaryLoss(
        n_samples=n_samples, system_radius=system_radius, t_max=t_max
    )

    loss_track = []
    w_track = []
    loss_names = ["pde", "av_in", "av_out", "init", "bnd"]

    def loss(G):
        # print(
        #     pde_loss(G=G, R=R),
        #     average_in_loss(G=G),
        #     average_out_loss(G=G),
        #     init_loss(G=G),
        #     boundary_loss(G=G),
        # )
        pde, av_in, av_out, init, bnd = (
            pde_loss(G=G, R=R),
            average_in_loss(G=G),
            average_out_loss(G=G),
            init_loss(G=G),
            boundary_loss(G=G),
        )
        loss_track.append((pde, av_in, av_out, init, bnd))
        total = pde + av_in + av_out + init + bnd
        total = total.detach()
        w_pde = pde.detach() / total
        w_av_in = av_in.detach() / total
        w_av_out = av_out.detach() / total
        w_init = init.detach() / total
        w_bnd = bnd.detach() / total * 0
        # print(w_pde, w_av_in, w_av_out, w_init, w_bnd)
        w_track.append((w_pde, w_av_in, w_av_out, w_init, w_bnd))
        weighted_sum = (
            w_pde * pde
            + w_av_in * av_in
            + w_av_out * av_out
            + w_init * init
            + w_bnd * bnd
        )
        return weighted_sum
        # return (
        #     pde_loss(G=G, R=R)
        #     + average_in_loss(G=G)
        #     + average_out_loss(G=G)
        #     + init_loss(G=G)
        #     + boundary_loss(G=G)
        # )

    G = G_MLP(
        nb_layer=params_pinns["n_layers"],
        hidden_layer=params_pinns["n_neurons_per_layer"],
    )
    optimizer = optim.Rprop(G.parameters())
    loss_list = []
    for it in tqdm(range(params_pinns["epoch"]), desc="Training process"):

        L = loss(G)
        loss_list.append(L.tolist())
        # L, L_total = cost(model, F_f, S_f, S_j, params["def_t"])
        # [loss[i].append(L_total[i].item()) for i in range(len(L_total))]
        # if var_R:
        #     loss[-1].append(model.R.item())

        # if L_total[0].item() <= min(loss[0]):
        #     model_opti = copy.deepcopy(model)

        L.backward()
        optimizer.step()
        # params["R"] = model_opti.R.item() * 10 ** (ordre_R + 1)

        # if update_progress != None and it % 1 == 0:
        #     update_progress(value=it + 1, best_R=params["R"], best_loss=min(loss[0]))
    # save(copy.deepcopy(model_opti), loss, params_pinns, params, output_path)
    # affichage(output_path)

    loss_track = torch.tensor(loss_track).detach()
    plt.figure()
    for i in range(loss_track.shape[1]):
        plt.plot(torch.tensor(loss_track)[:, i], label=loss_names[i])
    plt.plot(loss_list, label="Total")
    plt.yscale("log")
    plt.legend()
    plt.title("Losses")
    plt.xlabel("Iterations")

    w_track = torch.tensor(w_track).detach()
    plt.figure()
    for i in range(w_track.shape[1]):
        plt.plot(torch.tensor(w_track)[:, i], label=loss_names[i])
    plt.yscale("log")
    plt.legend()
    plt.title("Loss weights")
    plt.xlabel("Iterations")

    n_samples_r = 200
    n_samples_t = 100
    r = torch.linspace(0, system_radius, n_samples_r, requires_grad=True)
    times = torch.linspace(0, t_max, n_samples_t, requires_grad=True)
    m = torch.meshgrid([r, times])
    X = torch.cat([m[0].reshape(-1, 1), m[1].reshape(-1, 1)], dim=1)
    p_tensor = P_from_G(G(X), X).reshape(m[0].shape)

    print(p_tensor.shape)
    plt.figure()
    plt.imshow(
        p_tensor.detach().numpy(),
        extent=[
            0,
            system_radius,
            0,
            t_max,
        ],
        aspect="auto",
    )
    plt.colorbar()
    # plt.xticks(times.detach().numpy())
    # plt.yticks(r.detach().numpy())
    plt.show()
