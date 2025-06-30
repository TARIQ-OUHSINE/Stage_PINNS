import torch
import json, os, pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from pathlib import Path

from polpinn.tool import P, reload_model


# Dans save.py

def check(params, path):
    """
    Vérifie et crée l'arborescence nécessaire pour une exécution.
    Ne lève plus d'erreur si le dossier principal existe déjà.
    """
    # path est un objet Path, on n'a pas besoin de le convertir
    
    # S'assure que le dossier principal existe. Si non, le crée.
    path.mkdir(parents=True, exist_ok=True)
    
    # S'assure que les sous-dossiers nécessaires existent.
    (path / "Data").mkdir(exist_ok=True)
    (path / "Graphiques").mkdir(exist_ok=True)

    # La fonction n'a plus besoin de retourner quoi que ce soit.
    # L'ancienne logique avec 'raise FileExistsError' est supprimée.

"""def check(params, path):
    # <<<<<<< HEAD:code/polpinn/save.py
    # file_path = path / "Models"
    path.parent.mkdir(parents=True, exist_ok=True)
    # if not (os.path.exists(file_path)):
    # os.makedirs(file_path)
    # file_path_mod = file_path / params["name"]
    path.mkdir(exist_ok=False)
    # if os.path.exists(file_path_mod):
    # raise FileExistsError("Dossier déjà existant")
    # os.makedirs(file_path_mod)
    # os.makedirs(file_path_mod + r"\\Graphiques")
    # os.makedirs(file_path_mod + r"\\Data")
    (path / "Graphiques").mkdir()
    (path / "Data").mkdir() """


# =======
#     file_path = path + r"\\Models"
#     if not (os.path.exists(file_path)):
#         os.makedirs(file_path)
#     file_path_mod = file_path + f"\\" + params["name"]
#     if os.path.exists(file_path_mod):
#         return True
#     os.makedirs(file_path_mod)
#     os.makedirs(file_path_mod + r"\\Graphiques")
#     os.makedirs(file_path_mod + r"\\Data")
#     return False
# >>>>>>> a2aa7ebae1d95808b3e11395799e9d741f6d366f:save.py


def save(model, loss, params_PINNS, params, path):
    file_path = path / "Data"
    torch.save(model.state_dict(), file_path / "model.pth")

    with open(file_path / "loss.json", "w") as f:
        json.dump(loss, f)

    with open(file_path / "params.json", "w") as f:
        json.dump(params, f)

    with open(file_path / "params_PINNS.json", "w") as f:
        json.dump(params_PINNS, f)


import torch
import json, os, pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Assure-toi que ces importations sont bien en haut de ton fichier save.py
from polpinn.tool import P, reload_model

# ... (les autres fonctions de save.py comme check, save, etc.)

def affichage(path):
    """
    Génère et sauvegarde tous les graphiques de résultats pour une exécution.
    Version corrigée pour utiliser 'R_final_m' et gérer la normalisation.
    """
    model = reload_model(path=path)
    if isinstance(model, str):
        print(f"Erreur au rechargement du modèle : {model}")
        return model
        
    path = Path(path)
    if not (os.path.exists(path)):
        raise FileExistsError("Dossier n'existe pas")

    # --- Chargement de toutes les données nécessaires ---
    try:
        with open(path / "Data" / "loss.json", "r") as f:
            loss_data = json.load(f)
        with open(path / "Data" / "params.json", "r") as f:
            params = json.load(f)
        with open(path / "Data" / "params_PINNS.json", "r") as f:
            params_pinns = json.load(f)
        with open(path / "Data" / "S_f.pkl", "rb") as f:
            S_f = pickle.load(f)
        with open(path / "Data" / "S_j.pkl", "rb") as f:
            S_j = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Erreur: Fichier de résultats manquant. {e}")
        return

    # --- 1. Graphique de la fonction de coût ---
    var_R = params_pinns.get("var_R", False) # Utiliser .get pour éviter une KeyError
    
    # Définir les noms des courbes de loss en fonction des données sauvegardées
    nb_loss_curves = len(loss_data) - 1 if var_R else len(loss_data)
    loss_names = ["L_total", "L_solide", "L_bord", "L_ini", "L_fick", "L_fick_f", "L_fick_j"][:nb_loss_curves]

    fig1, ax1 = plt.subplots(1, 1, figsize=(13, 8))
    for i, name in enumerate(loss_names):
        ax1.plot(loss_data[i], label=name, linewidth=2)

    ax1.set_yscale("log")
    ax1.set_xlabel("Itérations", fontsize=16)
    ax1.set_ylabel("Coût (log)", fontsize=16)
    ax1.set_title("Évolution de la fonction de coût", fontsize=18, pad=15)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, which="both", ls="--")
    ax1.legend(fontsize=12)
    fig1.savefig(path / "Graphiques" / "loss.png")
    plt.close(fig1)

    # --- 2. Graphique de l'évolution du rayon (si applicable) ---
    if var_R:
        fig2, ax2 = plt.subplots(1, 1, figsize=(13, 8))
        rayon_history_norm = loss_data[-1]
        ordre_R = params.get('ordre_R', 0) # Sécurité si ordre_R n'est pas là
        rayon_history_m = [r * 10**(ordre_R + 1) for r in rayon_history_norm]
        
        ax2.plot(rayon_history_m, linewidth=2, color='green')
        ax2.set_xlabel("Itérations", fontsize=16)
        ax2.set_ylabel("Rayon estimé (m)", fontsize=16)
        ax2.set_title("Évolution du rayon R", fontsize=18, pad=15)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.grid(True, ls="--")
        fig2.savefig(path / "Graphiques" / "R_evolution.png")
        plt.close(fig2)

    # --- 3. Graphiques de comparaison des polarisations moyennes ---
    
    # Récupération et calcul des rayons (réel, final, normalisé)
    rayon_final_m = params['R_final_m']
    ordre_R = params['ordre_R']
    rayon_final_norm = rayon_final_m * 10**(-(ordre_R + 1))
    
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(26, 8))
    X_t = torch.linspace(0, params["def_t"], 1000).view(-1, 1)

    # Comparaison pour S_f (solide)
    X_R_norm = torch.full((1000, 1), rayon_final_norm)
    X_sf = torch.cat([X_R_norm, X_t], dim=1)
    
    G_values = model(X_sf).detach() * model.coeff_normal # Dénormaliser la sortie G
    ax3_1.plot(X_t.numpy(), S_f(X_t.numpy()), color="black", linestyle="--", label="S_f (données fittées)")
    ax3_1.plot(X_t.numpy(), G_values.numpy(), label="G(R,t) du modèle", linewidth=2)
    ax3_1.scatter(S_f.times, S_f.list_y, label="Données brutes S_f", color="red", s=40, zorder=5)
    ax3_1.set_xlabel("t (s)", fontsize=21)
    ax3_1.set_ylabel("Polarisation", fontsize=21)
    ax3_1.set_title("Polarisation moyenne du solide", fontsize=23, pad=15)
    ax3_1.legend(fontsize=17)
    ax3_1.grid(True)

    # Comparaison pour S_j (solvant)
    X_bord_norm = torch.full((1000, 1), rayon_final_norm + 1e-4) # Juste à l'extérieur
    X_sj = torch.cat([X_bord_norm, X_t], dim=1)
    X_sj.requires_grad_(True)
    
    P_values = P(model(X_sj), X_sj).detach() * model.coeff_normal # Dénormaliser P
    ax3_2.plot(X_t.numpy(), S_j(X_t.numpy()), color="black", linestyle="--", label="S_j (données fittées)")
    ax3_2.plot(X_t.numpy(), P_values.numpy(), label="P(R,t) du modèle", linewidth=2)
    ax3_2.scatter(S_j.times, S_j.list_y, label="Données brutes S_j", color="blue", s=40, zorder=5)
    ax3_2.set_xlabel("t (s)", fontsize=21)
    ax3_2.set_ylabel("Polarisation", fontsize=21)
    ax3_2.set_title("Polarisation à la frontière du solide", fontsize=23, pad=15)
    ax3_2.legend(fontsize=17)
    ax3_2.grid(True)

    fig3.savefig(path / "Graphiques" / "S_f_and_S_j_comparison.png")
    plt.close(fig3)

    # --- 4. Graphique de la polarisation ponctuelle (Colormap) ---
    nb_points = 100
    # Le modèle prend des entrées normalisées
    r_norm_np = np.linspace(0, rayon_final_norm, nb_points)
    t_np = np.linspace(0, params["def_t"], nb_points)
    
    r_grid_norm, t_grid = np.meshgrid(r_norm_np, t_np)
    X_grid_torch = torch.tensor(
        np.vstack([r_grid_norm.ravel(), t_grid.ravel()]).T, 
        dtype=torch.float32, 
        requires_grad=True
    )
    
    P_grid = P(model(X_grid_torch), X_grid_torch).detach() * model.coeff_normal
    P_grid_np = P_grid.numpy().reshape((nb_points, nb_points))

    # Pour l'affichage, on utilise les unités réelles (nm)
    r_display_nm = np.linspace(0, rayon_final_m * 1e9, nb_points)
    r_grid_display, t_grid_display = np.meshgrid(r_display_nm, t_np)

    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 8))
    contour = ax4.contourf(r_grid_display, t_grid_display, P_grid_np, 50, cmap="jet")
    
    cbar = fig4.colorbar(contour, label="P(r,t)")
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.yaxis.label.set_size(15)

    ax4.set_xlabel("r (nm)", fontsize=15)
    ax4.set_ylabel("t (s)", fontsize=15)
    ax4.tick_params(labelsize=13)
    ax4.set_title(
        f"Polarisation ponctuelle (R_final = {rayon_final_m * 1e9:.2f} nm)",
        fontsize=18,
        pad=15
    )
    fig4.savefig(path / "Graphiques" / "Polarisation_Ponctuelle.png")
    plt.close(fig4)

# Il faudra aussi corriger les fonctions 'cercle' et 'cercle_for_frontend' 
# de la même manière si tu les utilises.

def cercle(path, no_interaction=False):
    model = reload_model(path)
    if isinstance(model, str):
        return model
    with open(path / "Data" / "params.json", "r") as f:
        params = json.load(f)

    nb_point = 120
    def_t = params["def_t"]

    def P_for_colormap(r, t):
        t_array = np.full_like(r, t)
        X = np.vstack((r.ravel(), t_array.ravel())).T
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        P_ = P(model(X_tensor), X_tensor)
        P_ = P_.detach().numpy()
        P_ = P_.reshape(r.shape)
        return P_

    theta = np.linspace(0, 2 * np.pi, 100)
    r = np.linspace(0, params["R"] * 10**9, nb_point)
    r_for_RN = np.linspace(0, params["R"], nb_point)
    T, R = np.meshgrid(theta, r)
    T_for_RN, R_for_RN = np.meshgrid(theta, r_for_RN)

    colors = P_for_colormap(R_for_RN, 0)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    fig.set_size_inches(9, 7)
    plt.subplots_adjust(bottom=0.25)
    c = ax.pcolormesh(T, R, colors, cmap="jet", shading="auto")
    ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title(
        f"Polarisation ponctuelle d'une sphère de rayon:  {params['R']*10**9:.3e} nm",
        fontsize=20,
        pad=30,
    )

    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label("P(r, t)")
    cbar.ax.yaxis.label.set_size(15)
    cbar.ax.tick_params(labelsize=13)

    c.set_clim(0, params["P0_j"])

    ax_time = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    time_slider = Slider(ax_time, "Time", 0.0, def_t, valinit=0)

    time_text = ax.text(
        0.0,
        0.0,
        f"t = {0:.2f} s",
        fontsize=15,
        verticalalignment="top",
        horizontalalignment="center",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=1.0, edgecolor="white"),
    )

    def update(val):
        t = time_slider.val
        colors = P_for_colormap(R_for_RN, t)
        time_text.set_text(f"t = {t:.2f} s")
        c.set_array(colors.ravel())
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    if not no_interaction:
        plt.show()


def cercle_for_frontend(path):
    model = reload_model(path)
    path = Path(path)
    if isinstance(model, str):
        return model
    with open(path / "Data" / "params.json", "r") as f:
        params = json.load(f)

    nb_point = 120
    def_t = params["def_t"]

    def P_for_colormap(r, t):
        t_array = np.full_like(r, t)
        X = np.vstack((r.ravel(), t_array.ravel())).T
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        P_ = P(model(X_tensor), X_tensor)
        P_ = P_.detach().numpy()
        P_ = P_.reshape(r.shape)
        return P_

    theta = np.linspace(0, 2 * np.pi, 100)
    r = np.linspace(0, params["R"] * 10**9, nb_point)
    r_for_RN = np.linspace(0, params["R"], nb_point)
    T, R = np.meshgrid(theta, r)
    T_for_RN, R_for_RN = np.meshgrid(theta, r_for_RN)

    colors = P_for_colormap(R_for_RN, 0)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    fig.set_size_inches(9, 6)
    plt.subplots_adjust(bottom=0.25)
    c = ax.pcolormesh(T, R, colors, cmap="jet", shading="auto")
    ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title(
        f"Polarisation ponctuelle d'une sphère de rayon:  {params['R']*10**9:.3e} nm",
        fontsize=18,
        pad=30,
    )

    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label("P(r, t)")
    cbar.ax.yaxis.label.set_size(15)
    cbar.ax.tick_params(labelsize=13)

    c.set_clim(0, params["P0_j"])

    ax_time = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    time_slider = Slider(ax_time, "Time", 0.0, def_t, valinit=0)

    time_text = ax.text(
        0.0,
        0.0,
        f"t = {0:.2f} s",
        fontsize=15,
        verticalalignment="top",
        horizontalalignment="center",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=1.0, edgecolor="white"),
    )

    def update(val):
        t = time_slider.val
        colors = P_for_colormap(R_for_RN, t)
        time_text.set_text(f"t = {t:.2f} s")
        c.set_array(colors.ravel())
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    return [fig, ax]
