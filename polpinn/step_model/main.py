import torch, copy, shutil, os
import torch.optim as optim
from tqdm import tqdm
import copy, shutil
import matplotlib.pyplot as plt

from polpinn.simple_model.data import load_data, data_augmentation
from polpinn.simple_model.model import G_MLP, PointPolarization
from polpinn.step_model.losses import compute_loss
from polpinn.step_model.utils import step_function
from polpinn.utils import get_data_dir, get_output_dir


def run(name="Moyenne_homo_HyperP", seed=1234, update_progress=None):
    torch.manual_seed(seed)

    extra_radius_ratio = 1.09
    n_samples = 1000
    params_pinns = {
        "n_layers": 2,
        "n_neurons_per_layer": 32,
        "lr": 0.001,
        "lr_R": 0.0001,
        "epoch": 1000,
        "var_R": False,
    }

    # Load data
    data = load_data(data_path=get_data_dir() / name)
    params = data["params"]
    print(params)

    t_max = params["def_t"]
    rayon_ini = params["rayon_initialisation"]
    system_radius = params["R_bis"]

    normalisation_rayon = system_radius / 0.1
    rayon_ini = rayon_ini / normalisation_rayon
    system_radius = system_radius / normalisation_rayon
    R = torch.tensor(rayon_ini)

    T = step_function(threshold=rayon_ini, value_before=params["T_1"], value_after=params["T_B"])
    D = step_function(threshold=rayon_ini, value_before=params["D_f"] / normalisation_rayon ** 2, value_after=params["D_j"] / normalisation_rayon ** 2)
    P0 = step_function(threshold=rayon_ini, value_before=params["P0_f_normalized"], value_after=params["P0_j_normalized"])
    
    # T1 = params["T_1"]
    # P0 = params["P0_f_normalized"]
    # D_f = params["D_f"]

    # Normalize w.r.t. radius
    # normalisation_rayon = rayon_ini / 0.1
    # rayon_ini = rayon_ini / normalisation_rayon
    # D_f = D_f / normalisation_rayon ** 2


    # Initialize models
    print(params_pinns)
    model_G = G_MLP(
        nb_layer=params_pinns["n_layers"],
        hidden_layer=params_pinns["n_neurons_per_layer"],
    )
    model_P = PointPolarization(G=model_G)

    # Data augmentation
    data_aug = {}
    data_aug["average_in"] = data_augmentation(mono=False)
    data_aug["average_in"].fit(**data["average_in"])
    data_aug["average_out"] = data_augmentation(mono=False)
    data_aug["average_out"].fit(**data["average_out"])
    # t_data = torch.rand(n_samples) * t_max
    t_data = torch.linspace(0, t_max, n_samples)
    augmentated_data = {
        k: {"times": t_data, "values": data_aug[k](t_data)}
        for k in ("average_in", "average_out")
    }

    optimizer = optim.Rprop(model_G.parameters())
    losses = {}

    for it in tqdm(range(params_pinns["epoch"]), desc="Training process"):
        optimizer.zero_grad()

        losses_it = compute_loss(
            model_G=model_G,
            model_P=model_P,
            data_in=augmentated_data["average_in"],
            data_out=augmentated_data["average_out"],
            R=R,
            system_radius=system_radius,
            t_max=t_max,
            T=T,
            P0=P0,
            D=D,
            n_samples=n_samples,
            extra_radius_ratio=extra_radius_ratio
        )
        for k in losses_it:
            if k not in losses:
                losses[k] = torch.zeros(params_pinns["epoch"])
            losses[k][it] = losses_it[k]

        losses_it["total"].backward()
        optimizer.step()

    print("*" * 80)
    print("*" * 35, f'{losses["total"][-1].item():.2e}', "*" * 35)
    print("*" * 80)

    out_dir = get_output_dir() / 've_step'
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(1,2)
    plt.sca(ax[0])
    data_aug["average_in"].plot()
    ax[0].set_title('Data in')
    plt.sca(ax[1])
    data_aug["average_out"].plot()
    ax[1].set_title('Data out')
    plt.savefig(out_dir / f"{params['name']}_data.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(26, 8))
    for k in losses:
        ax.plot(losses[k].detach().numpy(), label=k)

    ax.set_yscale("log")
    ax.set_xlabel("itérations", fontsize=21)
    ax.set_ylabel("Coût", fontsize=21)
    ax.set_title("Coût en fonction des itérations", fontsize=23, pad=5)
    ax.tick_params(axis="x", labelsize=19)
    ax.tick_params(axis="y", labelsize=19)
    ax.grid(True)
    ax.legend(fontsize=17)
    plt.savefig(out_dir / f"{params['name']}_losses.pdf")

    X_r = torch.linspace(0, system_radius, 100)
    X_t = torch.linspace(0, params["def_t"], 100)
    X_r, X_t = torch.meshgrid([X_r, X_t], indexing='ij')
    X = torch.cat([X_r.reshape(-1, 1), X_t.reshape(-1, 1)], dim=1)

    p_tensor = model_P(X)

    plt.figure(figsize=(8, 6))
    plt.contourf(X_r*normalisation_rayon*1e9, X_t, p_tensor.detach().numpy().reshape(X_r.shape), 50, cmap="jet")

    cbar = plt.colorbar(label="P(r,t)")
    cbar.ax.yaxis.label.set_size(15)
    cbar.ax.tick_params(labelsize=13)

    plt.xlabel("r (nm)", fontsize=15)
    plt.ylabel("t (s)", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(
        f"Polarisation ponctuelle d'une sphère de rayon:  {params['rayon_initialisation']*10**9:.3f} nm",
        fontsize=16,
        pad=16,
        x=0.6,
    )
    plt.savefig(out_dir / f"{params['name']}_polarisation.pdf")


def run_all():
    for k in get_data_dir().glob('*'):
        print('#' * 80)
        print(k.stem)
        if k.is_dir():
            run(name=k.stem)

if __name__ == "__main__":
    # run()
    run_all()
    plt.show()

