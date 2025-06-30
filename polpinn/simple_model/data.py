from json import load
from math import isclose
from pandas import read_excel
import numpy as np
import torch
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from polpinn.utils import get_data_dir


def load_data(data_path=get_data_dir() / "Moyenne_homo_HyperP"):
    with open(data_path / "params.json", "r") as f:
        params = load(f)
    P0j = params["P0_j"]
    params["P0_f_normalized"] = params["P0_f"] / P0j
    params["P0_j_normalized"] = params["P0_j"] / P0j

    if isclose(params["R_bis"], 0):
        params["R_bis"] = 1e-06             #3 * params["rayon_initialisation"]
    if isclose(params["D_j"], 0):
        params["D_j"] = 5e-16
        params["rayon_initialisation"] = 2.5e-7

    catExcel = read_excel(data_path / ("S_f" + ".xlsx"))
    average_in = dict(
        times=torch.tensor(catExcel["t"], dtype=torch.float32),
        values=torch.tensor(catExcel["y"] / P0j, dtype=torch.float32),
    )
    catExcel = read_excel(data_path / ("S_j" + ".xlsx"))
    average_out = dict(
        times=torch.tensor(catExcel["t"], dtype=torch.float32),
        values=torch.tensor(catExcel["y"] / P0j, dtype=torch.float32),
    )

    return dict(params=params, average_in=average_in, average_out=average_out)


class data_augmentation:
    def __init__(self, mono=False) -> None:
        self.mono = mono
        self._times = None
        self._values = None

    def __call__(self, t):
        return self.C * (1 - np.exp(-((t / self.tau) ** self.beta)))

    def fit(self, times, values):
        self._times = times
        self._values = values
        self.tau, self.beta, self.C = 1.0, 1.0, max(values)

        def cost(params):
            if self.mono:
                self.tau, self.C = params
            else:
                self.tau, self.beta, self.C = params
            t = times.detach().numpy()
            y = values.detach().numpy()
            return (1 / len(t)) * np.sum((self(t) - y) ** 2)

        cost_history = []
        params_history = []

        def callback(params):
            cost_history.append(float(cost(params)))
            params_history.append(params.copy())

        iteration = 1000

        if self.mono:
            initial_params = [self.tau, self.C]

        else:
            initial_params = [self.tau, self.beta, self.C]

        self.params = np.array(initial_params, dtype=float)

        result = minimize(
            cost,
            self.params,
            method="L-BFGS-B",
            callback=callback,
            options={"maxiter": iteration},
        )
        best_params = params_history[np.argmin(cost_history)]
        if self.mono:
            self.tau, self.C = best_params
        else:
            self.tau, self.beta, self.C = best_params
        return self

    def plot(self):
        t, y = self._times, self._values
        plt.scatter(t, y, label="original data")
        t_range = np.linspace(0, torch.max(t), 100)
        plt.plot(t_range, self(t_range), color="r", label="augmentation")
        plt.legend()


if __name__ == "__main__":

    data = load_data()
    print(data)

    fig, axes = plt.subplots(1, 2)
    axes[0].scatter(
        data["average_in"]["times"], data["average_in"]["values"], label="data in"
    )
    axes[1].scatter(
        data["average_out"]["times"], data["average_out"]["values"], label="data out"
    )

    data_aug_in = data_augmentation(mono=False)
    data_aug_in.fit(**data["average_in"])

    data_aug_out_mono = data_augmentation(mono=True)
    data_aug_out_mono.fit(**data["average_out"])
    data_aug_out = data_augmentation(mono=False)
    data_aug_out.fit(**data["average_out"])

    t, y = data["average_in"]["times"], data["average_in"]["values"]
    t = np.linspace(0, torch.max(t), 100)
    axes[0].plot(t, data_aug_in(t), color="r", label="augmentation")
    axes[0].legend()
    t, z = data["average_out"]["times"], data["average_out"]["values"]
    t = np.linspace(0, torch.max(t), 100)
    axes[1].plot(t, data_aug_out_mono(t), color="r", label="augmentation mono")
    axes[1].plot(t, data_aug_out(t), "--", color="g", label="augmentation")
    axes[1].legend()

    plt.show()
