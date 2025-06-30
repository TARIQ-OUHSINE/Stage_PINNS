import torch, copy, shutil, os
import torch.optim as optim
from tqdm import tqdm
import copy, shutil

from polpinn.playground.losses import AverageInLoss, AverageOutLoss, InitLoss, PDELoss


def cost(model, model_P, data_in, data_out, R, t_max, T1, P0, D_f):
    n_samples = 1000
    L_solide = AverageInLoss(**data_in, radius=R)(G=model)
    # L_bord = AverageOutLoss(**data_out, radius=R + 0.009)(model_P)
    # L_ini = InitLoss(n_samples=n_samples, radius=R + 0.009)(model_P)
    L_bord = AverageOutLoss(**data_out, radius=R + 0.009e-7)(model_P)
    L_ini = InitLoss(n_samples=n_samples, radius=R + 0.009e-7)(model_P)
    L_fick_f = PDELoss(
        D=D_f,
        T=T1,
        P0=P0,
        radius=R,
        t_max=t_max,
        n_samples=n_samples,
    )(P=model_P)
    sum = L_solide + L_bord + L_ini + L_fick_f
    return sum, [sum, L_solide, L_bord, L_ini, L_fick_f]


def run(
    seed=1234,
    update_progress=None,
):
    from polpinn.playground.data import load_data, data_augmentation

    data = load_data()
    params = data["params"]
    print(params)

    rayon_ini = params["rayon_initialisation"]
    D_f = params["D_f"]
    print(rayon_ini, D_f)

    torch.manual_seed(seed)
    R = torch.tensor(rayon_ini)
    from polpinn.playground.model import G_MLP, PointPolarization

    params_pinns = {
        "n_layers": 2,
        "n_neurons_per_layer": 32,
        "lr": 0.001,
        "lr_R": 0.0001,
        "epoch": 200,
        "var_R": False,
    }
    print(params_pinns)
    model = G_MLP(
        nb_layer=params_pinns["n_layers"],
        hidden_layer=params_pinns["n_neurons_per_layer"],
    )

    model_P = PointPolarization(G=model)

    n_samples = 1000
    T1 = params["T_1"]
    P0 = params["P0_f_normalized"]
    t_max = params["def_t"]
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
    # S_f = data_augmentation(mono=False).fit(**data["average_in"])
    # S_j = data_augmentation(mono=False).fit(**data["average_out"])

    optimizer = optim.Rprop(model.parameters())
    loss = [[] for _ in range(5)]

    for it in tqdm(range(params_pinns["epoch"]), desc="Training process"):
        optimizer.zero_grad()

        L, L_total = cost(
            model,
            model_P,
            data_in=augmentated_data["average_in"],
            data_out=augmentated_data["average_out"],
            R=R,
            t_max=t_max,
            T1=T1,
            P0=P0,
            D_f=D_f,
        )
        [loss[i].append(L_total[i].item()) for i in range(len(L_total))]

        if L_total[0].item() <= min(loss[0]):
            model_opti = copy.deepcopy(model)

        L.backward()
        optimizer.step()

        if update_progress != None and it % 1 == 0:
            update_progress(value=it + 1, best_R=params["R"], best_loss=min(loss[0]))
    print("*" * 80)
    print("*" * 10, L.item(), "*" * 10)
    print("*" * 80)
    import matplotlib.pyplot as plt

    fig, (ax1) = plt.subplots(1, 1, figsize=(26, 8))
    loss_name = [
        "L",
        "L_solide",
        "L_bord",
        "L_ini",
        "L_fick_f",
    ]
    for i in range(len(loss_name)):
        (line,) = ax1.plot(
            range(0, len(loss[i])), loss[i], label=loss_name[i], linewidth=2
        )

    ax1.set_yscale("log")
    ax1.set_xlabel("itérations", fontsize=21)
    ax1.set_ylabel("Coût", fontsize=21)
    ax1.set_title("Coût en fonction des itérations", fontsize=23, pad=5)
    ax1.tick_params(axis="x", labelsize=19)
    ax1.tick_params(axis="y", labelsize=19)
    ax1.grid(True)
    ax1.legend(fontsize=17)

    # X_r = torch.linspace(0, params["rayon_initialisation"] * 10**9, 100)
    X_r = torch.linspace(0, params["rayon_initialisation"], 100)
    X_t = torch.linspace(0, params["def_t"], 100)
    X_r, X_t = torch.meshgrid([X_r, X_t], indexing='ij')
    X = torch.cat([X_r.reshape(-1, 1), X_t.reshape(-1, 1)], dim=1)

    p_tensor = model_P(X)

    plt.figure(figsize=(8, 6))
    plt.contourf(X_r, X_t, p_tensor.detach().numpy().reshape(X_r.shape), 50, cmap="jet")

    cbar = plt.colorbar(label="P(r,t)")
    cbar.ax.yaxis.label.set_size(15)
    cbar.ax.tick_params(labelsize=13)

    plt.xlabel("r (nm)", fontsize=15)
    plt.ylabel("t (s)", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(
        f"Polarisation ponctuelle d'une sphère de rayon:  {params["rayon_initialisation"]*10**9:.3e} nm",
        fontsize=16,
        pad=16,
        x=0.6,
    )

    plt.show()


if __name__ == "__main__":
    run()
