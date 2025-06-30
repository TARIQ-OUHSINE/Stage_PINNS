# Importations nécessaires au début de votre fichier
import json
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from polpinn.tool import P, reload_model
# Assurez-vous que vos fonctions `reload_model` et `P` sont accessibles
# from polpinn.PINN import reload_model, P 


def affichage(path):
    # Le début reste identique
    model = reload_model(path=path)
    if isinstance(model, str):
        return model
    path = Path(path)
    if not (os.path.exists(path)):
        raise FileExistsError("Dossier n'existe pas")

    # Les vérifications de fichiers restent identiques
    if not (os.path.isfile(path / "Data" / "model.pth")):
        raise FileExistsError("model.pth n'existe pas")
    # ... (toutes vos autres vérifications de fichiers)
    if not (os.path.isfile(path / "Data" / "S_j_mono.pkl")):
        raise FileExistsError("S_j_mono.pkl n'existe pas")

    # Le chargement des données reste identique
    with open(path / "Data" / "loss.json", "r") as f:
        loss = json.load(f)
    with open(path / "Data" / "params.json", "r") as f:
        params = json.load(f)
    with open(path / "Data" / "params_PINNS.json", "r") as f:
        params_PINNS = json.load(f)
    with open(path / "Data" / "S_f.pkl", "rb") as f:
        S_f = pickle.load(f)
    with open(path / "Data" / "S_j.pkl", "rb") as f:
        S_j = pickle.load(f)
    with open(path / "Data" / "S_j_mono.pkl", "rb") as f:
        S_j_mono = pickle.load(f)

    # ### AJOUTÉ 1/3 : "Traduire" les nouvelles données pour l'ancien code ###
    # L'ancien code a besoin de params['R'] (normalisé). On le calcule à partir
    # des nouvelles données params['R_final_m'] et params['ordre_R'].
    # Le reste du code pourra maintenant utiliser params['R'] sans modification.
    if 'R_final_m' in params and 'ordre_R' in params:
        params['R'] = params['R_final_m'] * (10**(-(params['ordre_R'] + 1)))
    else:
        # Sécurité si les nouvelles clés ne sont pas trouvées
        raise KeyError("Le fichier params.json doit contenir 'R_final_m' et 'ordre_R'")


    # Le bloc de tracé de la loss et du rayon reste structurellement identique
    if params_PINNS["var_R"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 8))
        loss_name = ["L", "L_solide", "L_bord", "L_ini", "L_fick_f"]
        for i in range(len(loss_name)):
            ax1.plot(loss[i], label=loss_name[i], linewidth=2)

        ax1.set_yscale("log")
        ax1.set_xlabel("itérations", fontsize=21)
        ax1.set_ylabel("Coût", fontsize=21)
        ax1.set_title("Coût en fonction des itérations", fontsize=23, pad=5)
        ax1.tick_params(labelsize=19)
        ax1.grid(True)
        ax1.legend(fontsize=17)

        # ### MODIFIÉ 2/3 : Rendre le graphique du rayon lisible ###
        # Au lieu de tracer la valeur normalisée, on la convertit en mètres
        # pour un affichage physiquement correct, tout en gardant le style.
        ordre_R = params.get('ordre_R', 0)
        rayon_history_m = [r * 10**(ordre_R + 1) for r in loss[-1]]
        ax2.plot(rayon_history_m, linewidth=2)
        ax2.set_xlabel("itérations", fontsize=21)
        ax2.set_ylabel("R (m)", fontsize=21) # Label mis à jour pour être correct
        ax2.set_title("R en fonction des itérations", fontsize=23, pad=5)
        ax2.tick_params(labelsize=19)
        ax2.grid(True)
        fig.savefig(path / "Graphiques" / "loss_and_R.png")
    else:
        # Cette partie ne change pas
        fig, ax1 = plt.subplots(1, 1, figsize=(26, 8))
        loss_name = ["L", "L_solide", "L_bord", "L_ini", "L_fick_f"]
        for i in range(len(loss_name)):
            ax1.plot(loss[i], label=loss_name[i], linewidth=2)
        ax1.set_yscale("log")
        ax1.set_xlabel("itérations", fontsize=21)
        ax1.set_ylabel("Coût", fontsize=21)
        ax1.set_title("Coût en fonction des itérations", fontsize=23, pad=5)
        ax1.tick_params(labelsize=19)
        ax1.grid(True)
        ax1.legend(fontsize=17)
        fig.savefig(path / "Graphiques" / "loss.png")
    plt.close()


    # Le bloc de comparaison des polarisations reste structurellement identique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 8))
    X_t = torch.linspace(0, params["def_t"], 1000).view(-1, 1)
    # On utilise le params['R'] qu'on a calculé au début
    X_R = torch.full((1000, 1), params["R"]) 
    X = torch.cat([X_R, X_t], dim=1)
    X.requires_grad_(True)

    ax1.plot(X_t.detach().numpy(), S_f(X_t.numpy()), color="black", linestyle="--", label="S_f")
    # ### MODIFIÉ 3/3 : Appliquer la dénormalisation de la sortie du modèle ###
    # C'est le changement le plus critique pour la validité des résultats.
    prediction_sf = (model(X) * model.coeff_normal).view(-1).detach().numpy()
    ax1.plot(X_t.view(-1).detach().numpy(), prediction_sf, label="S_f décrit par le modèle", linewidth=2)
    X_times = [i for i in S_f.times if i <= params["def_t"]]
    ax1.scatter(X_times, S_f.list_y[: len(X_times)], label="y_i", color="red", s=40)
    ax1.set_xlabel("t (s)", fontsize=21)
    ax1.set_ylabel("S_f(t)", fontsize=21)
    ax1.set_title("Polarisation moyenne du solide", fontsize=23, pad=5)
    ax1.tick_params(labelsize=19)
    ax1.grid(True)
    ax1.legend(fontsize=17)

    X = torch.cat([X_R + 0.009 * 10 ** (params["ordre_R"] + 1), X_t], dim=1)
    X.requires_grad_(True)
    ax2.plot(X_t.detach().numpy(), S_j(X_t.numpy()), color="black", linestyle="--", label="S_j")
    S_j_mono_values = S_j_mono(X_t.numpy())
    loss_strecth_mono = np.mean(np.abs(S_j(X_t.numpy()) - S_j_mono_values))
    ax2.plot(X_t.detach().numpy(), S_j_mono_values, color="green", linestyle="--", label=f"S_j_mono, loss={loss_strecth_mono:.2e}")
    # ### MODIFIÉ 3/3 (bis) : Appliquer la dénormalisation ici aussi ###
    prediction_sj = (P(model(X), X) * model.coeff_normal).view(-1).detach().numpy()
    ax2.plot(X_t.view(-1).detach().numpy(), prediction_sj, label="P(r>R,t) décrit par le modèle", linewidth=2)
    X_times = [i for i in S_j.times if i <= params["def_t"]]
    ax2.scatter(X_times, S_j.list_y[: len(X_times)], label="z_i", color="red", s=40)
    ax2.set_xlabel("t (s)", fontsize=21)
    ax2.set_ylabel("S_j(t) = P(r>R,t)", fontsize=21)
    ax2.set_title("Polarisation moyenne du solvant", fontsize=23, pad=5)
    ax2.tick_params(labelsize=19)
    ax2.grid(True)
    ax2.legend(fontsize=17)
    fig.savefig(path / "Graphiques" / "S_f_and_S_j.png")
    plt.close()


    # Le bloc Colormap reste structurellement identique
    X_t_grid = np.linspace(0, params["def_t"], 100)
    X_r_grid_norm = np.linspace(0, params["R"], 100) # Utilise le R normalisé calculé
    X_r_mesh, X_t_mesh = np.meshgrid(X_r_grid_norm, X_t_grid)
    X = np.vstack([X_r_mesh.ravel(), X_t_mesh.ravel()]).T
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    # ### MODIFIÉ 3/3 (ter) : Appliquer la dénormalisation pour la colormap ###
    P_ = P(model(X), X) * model.coeff_normal
    P_ = P_.detach().numpy().reshape(X_r_mesh.shape)
    np.save(path / "Data" / "P.npy", P_)

    # Affichage en nm, mais en utilisant R_final_m pour être correct
    X_r_display = np.linspace(0, params["R_final_m"] * 1e9, 100)
    X_r_display_mesh, X_t_display_mesh = np.meshgrid(X_r_display, X_t_grid)
    np.save(path / "Data" / "(r, t).npy", (X_r_display_mesh, X_t_display_mesh))

    plt.figure(figsize=(8, 6))
    plt.contourf(X_r_display_mesh, X_t_display_mesh, P_, 50, cmap="jet")
    cbar = plt.colorbar(label="P(r,t)")
    cbar.ax.yaxis.label.set_size(15)
    cbar.ax.tick_params(labelsize=13)
    plt.xlabel("r (nm)", fontsize=15)
    plt.ylabel("t (s)", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # ### MODIFIÉ : Utiliser la bonne valeur pour le titre ###
    plt.title(
        f"Polarisation ponctuelle d'une sphère de rayon:  {params['R_final_m']*1e9:.3e} nm",
        fontsize=16, pad=16, x=0.6,
    )
    plt.savefig(path / "Graphiques" / "Polarisation_Ponctuelle.png")
    plt.close()