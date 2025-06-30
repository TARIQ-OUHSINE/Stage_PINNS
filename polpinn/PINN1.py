import torch, copy, shutil, os
import torch.optim as optim
from tqdm import tqdm
import copy, shutil

from polpinn.tool import (
    Fick,
    normalisation,
    data_augmentation,
    Physics_informed_nn,
    P,
    hypo,
)
from polpinn.save import check, save, affichage


def cost(model, F_f, S_f, S_j, def_t):
    R = model.R
    nb_r = 30
    nb_t = def_t * 50

    X_R = R.repeat(nb_t, 1).view(-1, 1)
    X_t = torch.linspace(0, def_t, nb_t).view(-1, 1)

    X_solide = torch.cat([X_R, X_t], dim=1)
    X_solide.requires_grad_(True)

    L_solide = torch.mean(torch.square(model(X_solide) - S_f(X_t) / model.coeff_normal))

    X_bord = torch.cat([X_R + 0.009, X_t], dim=1)
    X_bord.requires_grad_(True)

    L_bord = torch.mean(
        torch.square(P(model(X_bord), X_bord) - S_j(X_t) / model.coeff_normal)
    )

    X_r = torch.linspace(0, R + 0.009, nb_t).view(-1, 1)
    X_t_0 = torch.zeros((nb_t, 1)).view(-1, 1)
    X_ini = torch.cat([X_r, X_t_0], dim=1)
    X_ini.requires_grad_(True)

    L_ini = torch.mean(torch.square(P(model(X_ini), X_ini)))

    X_r_f = torch.linspace(0, R, nb_r).view(-1, 1)
    X_t = torch.linspace(0, def_t, nb_r).view(-1, 1)

    grid_r_f, grid_t_f = torch.meshgrid(X_r_f.squeeze(), X_t.squeeze(), indexing="ij")
    X_f = torch.stack([grid_r_f.flatten(), grid_t_f.flatten()], dim=1)
    X_f.requires_grad_(True)

    L_fick_f = torch.mean(torch.square(F_f(P(model(X_f), X_f), X_f)))

    sum = L_solide + L_bord + L_ini + L_fick_f
    gamma_solide = L_solide / sum
    gamma_bord = L_bord / sum
    gamma_ini = L_ini / sum
    gamma_fick_f = L_fick_f / sum

    return (
        gamma_solide * L_solide
        + gamma_bord * L_bord
        + gamma_ini * L_ini
        + gamma_fick_f * L_fick_f,
        [sum, L_solide, L_bord, L_ini, L_fick_f],
    )


def run(
    params_pinns: dict,
    params: dict,
    output_path,
    data_path,
    seed=1234,
    update_progress=None,
    no_interaction=False,
    no_gui=False,
):
    check(params=params, path=output_path)
    coeff_normal = params["P0_j"]
    var_R = params_pinns["var_R"]
    rayon_ini, D_f, ordre_R = normalisation(
        params["rayon_initialisation"], params["D_f"]
    )
    params["ordre_R"] = ordre_R
    print(rayon_ini)

    torch.manual_seed(seed)
    model = Physics_informed_nn(
        nb_layer=params_pinns["nb_hidden_layer"],
        hidden_layer=params_pinns["nb_hidden_perceptron"],
        rayon_ini=rayon_ini,
        coeff_normal=coeff_normal,
        var_R=var_R,
    )

    F_f = Fick(D_f, params["T_1"], params["P0_f"] / coeff_normal)
    S_f = data_augmentation(data_path, output_path, "S_f", params["name"])
    S_j = data_augmentation(data_path, output_path, "S_j", params["name"])
    S_j_mono = data_augmentation(
        data_path, output_path, "S_j", params["name"], mono=True
    )

    # if not (os.path.isfile(path + r"\\S_f.xlsx")):
    #     return "S_f.xlsx n'existe pas"
    # if not (os.path.isfile(path + r"\\S_j.xlsx")):
    #     return "S_j.xlsx n'existe pas"
    # if check(params=params, path=path):
    #     return "Dossier déjà existant"

    if hypo(
        S_f,
        S_j,
        S_j_mono,
        output_path / "Graphiques" / "S_f_and_S_j.png",
        no_interaction=no_interaction,
        no_gui=no_gui,
    ):
        shutil.rmtree(output_path)
        return

    if var_R:
        params_without_R = [
            param for name, param in model.named_parameters() if name != "R"
        ]
        optimizer = optim.Adam(
            [{"params": params_without_R}, {"params": model.R, "lr": 0}],
            lr=params_pinns["lr"],
        )
        loss = [[] for _ in range(6)]

    else:
        optimizer = optim.Rprop(model.parameters())
        loss = [[] for _ in range(5)]

    for it in tqdm(range(params_pinns["epoch"]), desc="Training process"):
        if it == 3000 and var_R:
            params_without_R = [
                param for name, param in model.named_parameters() if name != "R"
            ]
            optimizer = optim.Adam(
                [
                    {"params": params_without_R},
                    {"params": model.R, "lr": params_pinns["lr_R"]},
                ],
                lr=params_pinns["lr"],
            )
        elif it == 3000 and not (var_R):
            optimizer = optim.Adam(model.parameters())
        optimizer.zero_grad()

        L, L_total = cost(model, F_f, S_f, S_j, params["def_t"])
        [loss[i].append(L_total[i].item()) for i in range(len(L_total))]
        if var_R:
            loss[-1].append(model.R.item())

        if L_total[0].item() <= min(loss[0]):
            model_opti = copy.deepcopy(model)

        L.backward()
        optimizer.step()
        params["R"] = model_opti.R.item() * 10 ** (ordre_R + 1)

        if update_progress != None and it % 1 == 0:
            update_progress(value=it + 1, best_R=params["R"], best_loss=min(loss[0]))
    print(len(loss), len(loss[0]))
    save(copy.deepcopy(model_opti), loss, params_pinns, params, output_path)
    affichage(output_path)
