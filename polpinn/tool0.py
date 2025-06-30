from platform import system
from typing import Any


import torch, math, pickle, json, os
import torch.nn as nn
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import messagebox
from scipy.optimize import minimize
from pathlib import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from polpinn.utils import get_data_dir


class Physics_informed_nn(nn.Module):
    def __init__(
        self,
        nb_layer: int,
        hidden_layer: int,
        rayon_ini: float,
        coeff_normal: float,
        var_R: bool,
    ):
        super(Physics_informed_nn, self).__init__()
        self.coeff_normal = coeff_normal
        self.fc_int = nn.Linear(2, hidden_layer)
        self.fc = nn.ModuleList(
            [nn.Linear(hidden_layer, hidden_layer) for _ in range(nb_layer)]
        )
        self.fc_out = nn.Linear(hidden_layer, 1)
        if var_R:
            self.R = nn.Parameter(
                torch.tensor(float(rayon_ini), dtype=torch.float32, requires_grad=True)
            )
        else:
            self.R = torch.tensor(rayon_ini)

    def forward(self, x):
        x = torch.tanh(self.fc_int(x))
        for fc in self.fc:
            x = torch.tanh(fc(x))
        x = torch.sigmoid(self.fc_out(x))
        return x


class Fick:
    def __init__(self, D, T, P_0) -> None:
        self.D = D
        self.T = T
        self.P_0 = P_0

    def __call__(self, P, X, *args: Any, **kwds: Any) -> Any:
        X.requires_grad_(True)
        X_r = X[:, 0].view(-1, 1)
        dP_d = torch.autograd.grad(
            P, X, grad_outputs=torch.ones_like(P), create_graph=True
        )[0]
        dP_dr = dP_d[:, 0].view(-1, 1)
        dP_dt = dP_d[:, 1].view(-1, 1)
        dP_dd = torch.autograd.grad(
            dP_d,
            X,
            grad_outputs=torch.ones_like(dP_d),
            create_graph=True,
        )[0]
        dP_drr = dP_dd[:, 0].view(-1, 1)
        return (
            X_r * dP_dt
            - self.D * (X_r * dP_drr + 2 * dP_dr)
            + X_r * ((P - self.P_0) / self.T)
        )


def P(G, X, *args: Any, **kwds: Any) -> Any:
    X.requires_grad_(True)
    X_r = X[:, 0].view(-1, 1)
    dG_d = torch.autograd.grad(
        G, X, grad_outputs=torch.ones_like(G), create_graph=True
    )[0]
    dG_dr = dG_d[:, 0].view(-1, 1)
    P_f_tensor = (X_r / 3) * dG_dr + G
    return P_f_tensor


def normalisation(R, D):
    ordre_R = math.floor(math.log10(abs(R)))
    D = D / (10 ** (ordre_R + 1)) ** 2
    return (
        float(format(R * 10 ** (-(ordre_R + 1)), f".{2}e")),
        float(format(D, f".{4}e")),
        ordre_R,
    )


class data_augmentation:
    def __init__(self, data_path, output_path, S, name, mono=False) -> None:
        self.mono = mono
        self.data_path = data_path
        self.output_path = output_path
        self.S = S
        self.name = name
        catExcel = pd.read_excel(data_path / (self.S + ".xlsx"))

        self.times = np.array(catExcel["t"])
        self.list_y = np.array(catExcel["y"])

        self.tau, self.beta, self.C = 1.0, 1.0, max(self.list_y)
        best_params, self.min_loss = self.run()

        if self.mono:
            self.tau, self.C = best_params
            path = self.output_path / "Data" / (self.S + "_mono.pkl")
        else:
            self.tau, self.beta, self.C = best_params
            path = self.output_path / "Data" / (self.S + ".pkl")

        with open(path, "wb") as file:
            pickle.dump(self, file)

    def __call__(self, t):
        return self.C * (1 - np.exp(-((t / self.tau) ** self.beta)))

    def cost(self, params):
        if self.mono:
            self.tau, self.C = params
        else:
            self.tau, self.beta, self.C = params
        return (1 / len(self.times)) * np.sum((self(self.times) - self.list_y) ** 2)

    def callback(self, params):
        self.params = params
        self.cost_history.append(float(self.cost(params)))
        self.params_history.append(params.copy())

    def run(self):
        iteration = 1000

        if self.mono:
            initial_params = [self.tau, self.C]

        else:
            initial_params = [self.tau, self.beta, self.C]

        self.params = np.array(initial_params, dtype=float)

        self.cost_history = []
        self.params_history = []

        result = minimize(
            self.cost,
            self.params,
            method="L-BFGS-B",
            callback=self.callback,
            options={"maxiter": iteration},
        )
        self.params = result.x
        best_params = self.params_history[np.argmin(self.cost_history)]
        return best_params, min(self.cost_history)


def ask_confirmation():
    root_y_or_n = tk.Tk()
    root_y_or_n.withdraw()
    result = messagebox.askyesno("Confirmation", "Do you want to continue ?")
    root_y_or_n.destroy()
    return result


def hypo(S_f, S_j, S_j_mono, path, no_interaction, no_gui=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 8))
    min_times = min(S_f.times[-1], S_j.times[-1])
    X_t = torch.linspace(0, min_times, 1000).view(-1, 1)

    ax1.plot(
        X_t.view(-1).detach().numpy(),
        S_f(X_t).view(-1).detach().numpy(),
        color="black",
        linestyle="--",
        label="S_f",
    )
    X_times = [i for i in S_f.times if i <= min_times]
    ax1.scatter(X_times, S_f.list_y[: len(X_times)], label="y_i", color="red", s=40)
    ax1.set_xlabel("t (s)", fontsize=21)
    ax1.set_ylabel("S_f(t)", fontsize=21)
    ax1.set_title("Polarisation moyenne du solide", fontsize=23, pad=5)
    ax1.tick_params(axis="x", labelsize=19)
    ax1.tick_params(axis="y", labelsize=19)
    ax1.grid(True)
    ax1.legend(fontsize=17)

    S_j_values = S_j(X_t)
    ax2.plot(
        X_t.view(-1).detach().numpy(),
        S_j_values.view(-1).detach().numpy(),
        color="black",
        linestyle="--",
        label="S_j",
    )
    S_j_mono_values = S_j_mono(X_t)
    loss_strecth_mono = torch.mean(torch.abs(S_j_values - S_j_mono_values))
    ax2.plot(
        X_t.view(-1).detach().numpy(),
        S_j_mono_values.view(-1).detach().numpy(),
        color="green",
        linestyle="--",
        label=f"S_j_mono, loss={loss_strecth_mono.item():.2e}",
    )
    X_times = [i for i in S_j.times if i <= min_times]
    ax2.scatter(X_times, S_j.list_y[: len(X_times)], label="z_i", color="red", s=40)
    ax2.set_xlabel("t (s)", fontsize=21)
    ax2.set_ylabel("S_j(t) = P(R,t)", fontsize=21)
    ax2.set_title("Polarisation moyenne du solvant", fontsize=23, pad=5)
    ax2.tick_params(axis="x", labelsize=19)
    ax2.tick_params(axis="y", labelsize=19)
    ax2.grid(True)
    ax2.legend(fontsize=17)
    plt.savefig(path)
    if not no_interaction and no_gui:
        plt.show()

    if no_gui:
        if no_interaction:
            result = True
        else:
            result = input("Voulez-vous continuer? [y/n, default=y]")
            result = result in ("Y", "y", "")
    else:

        if system() == "Darwin":
            result = True
        else:
            result = None

            def on_closing():
                nonlocal result
                root_fig.destroy()
                result = ask_confirmation()
                root_fig.quit()

            root_fig = tk.Tk()
            root_fig.title("S_f&S_j")
            root_fig.state("zoomed")

            canvas = FigureCanvasTkAgg(fig, master=root_fig)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

            root_fig.protocol("WM_DELETE_WINDOW", on_closing)

            root_fig.mainloop()

    return not (result)


class PINNS_reload(nn.Module):
    def __init__(
        self,
        nb_layer: int,
        hidden_layer: int,
        coeff_normal: float,
        var_R: bool,
        ordre_R: int,
    ):
        super(PINNS_reload, self).__init__()
        self.coeff_normal = coeff_normal
        self.ordre_R = ordre_R
        self.fc_int = nn.Linear(2, hidden_layer)
        self.fc = nn.ModuleList(
            [nn.Linear(hidden_layer, hidden_layer) for _ in range(nb_layer)]
        )
        self.fc_out = nn.Linear(hidden_layer, 1)
        if var_R:
            self.R = nn.Parameter(
                torch.tensor(0, dtype=torch.float32, requires_grad=True)
            )

    def forward(self, x):
        X_r = x[:, 0].view(-1, 1)
        X_t = x[:, 1].view(-1, 1)

        X_r = X_r * 10 ** (-self.ordre_R - 1)

        x = torch.cat([X_r, X_t], dim=1)
        x = torch.tanh(self.fc_int(x))
        for fc in self.fc:
            x = torch.tanh(fc(x))
        x = torch.sigmoid(self.fc_out(x))
        return x * self.coeff_normal


def reload_model(path):
    path = Path(path)
    if not (os.path.exists(path)):
        return "Dossier n'existe pas"

    if not (os.path.isfile(path / "Data" / "model.pth")):
        return "model.pth n'existe pas"

    if not (os.path.isfile(path / "Data" / "params.json")):
        raise FileExistsError("params.json n'existe pas")

    if not (os.path.isfile(path / "Data" / "params_PINNS.json")):
        raise FileExistsError("params_PINNS.json n'existe pas")

    with open(path / "Data" / "params.json", "r") as f:
        params = json.load(f)

    with open(path / "Data" / "params_PINNS.json", "r") as f:
        params_PINNS = json.load(f)

    model = PINNS_reload(
        nb_layer=params_PINNS["nb_hidden_layer"],
        hidden_layer=params_PINNS["nb_hidden_perceptron"],
        coeff_normal=params["P0_j"],
        var_R=params_PINNS["var_R"],
        ordre_R=params["ordre_R"],
    )
    model.load_state_dict(torch.load(path / "Data" / "model.pth"))
    model.eval()
    return model
