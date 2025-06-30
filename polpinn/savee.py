# save.py (version complète et corrigée pour l'Axe 1)

import torch
import json
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from pathlib import Path

# On importe la nouvelle fonction de reload depuis tool.py
from polpinn.tool import reload_model_Axe1


def check(path: Path):
    """
    Crée l'arborescence de dossiers nécessaire pour une expérience.
    Crée les dossiers s'ils n'existent pas, ne fait rien s'ils existent déjà.
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / "Graphiques").mkdir(exist_ok=True)
    (path / "Data").mkdir(exist_ok=True)


def save(model: torch.nn.Module, loss: list, params_PINNS: dict, params: dict, path: Path):
    """
    Sauvegarde le modèle, l'historique de la loss et les dictionnaires de paramètres.
    """
    file_path = path / "Data"
    # On sauvegarde le state_dict du modèle optimisé
    torch.save(model.state_dict(), file_path / "model.pth")

    with open(file_path / "loss.json", "w") as f:
        json.dump(loss, f)

    with open(file_path / "params.json", "w") as f:
        json.dump(params, f, indent=4)

    with open(file_path / "params_PINNS.json", "w") as f:
        json.dump(params_PINNS, f, indent=4)


def affichage(path: Path):
    """
    Génère et sauvegarde toutes les figures de résultats pour une expérience de type Axe 1.
    """
    path = Path(path)
    # On utilise la nouvelle fonction de reload
    model = reload_model_Axe1(path=path)
    if isinstance(model, str):
        print(f"Erreur lors du chargement du modèle: {model}")
        return model

    # Chargement des metadonnées de l'expérience
    with open(path / "Data" / "loss.json", "r") as f:
        loss = json.load(f)
    with open(path / "Data" / "params.json", "r") as f:
        params = json.load(f)
    with open(path / "Data" / "params_PINNS.json", "r") as f:
        params_pinns = json.load(f)
    with open(path / "Data" / "S_f.pkl", "rb") as f:
        S_f = pickle.load(f)

    # --- 1. Graphique de la Loss (et du Rayon si variable) ---
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))
    loss_names = ["L_total", "L_fick", "L_solide", "L_ini"]
    for i in range(len(loss_names)):
        ax1.plot(loss[i], label=loss_names[i])
    
    ax1.set_yscale('log')
    ax1.set_title(f'Évolution de la Loss - Expérience: {params.get("name", "")}', fontsize=16)
    ax1.set_xlabel('Itérations', fontsize=12)
    ax1.set_ylabel('Valeur de la Loss (log)', fontsize=12)
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    
    # Si R est variable, on ajoute un deuxième axe Y pour son évolution
    if params_pinns.get("var_R", False) and len(loss) > len(loss_names):
        ax2 = ax1.twinx()
        final_R_estimé_nm = params.get("R_final_estimé", 0) * 1e9
        # Le rayon est déjà dé-normalisé dans l'historique de loss de PINN.py
        rayon_history_nm = np.array(loss[-1]) * 10**(params.get("ordre_R", -7)) * 1e9
        ax2.plot(rayon_history_nm, color='red', linestyle='--', label=f'Rayon (nm) (final: {final_R_estimé_nm:.1f} nm)')
        ax2.set_ylabel('Rayon estimé (nm)', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(path / "Graphiques" / "loss_evolution.png")
    plt.close(fig)

    # --- 2. Graphique de la Polarisation Moyenne du Solide S_f ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    t_plot = torch.linspace(0, params["def_t"], 200).view(-1, 1)
    
    # Données cibles
    ax.plot(t_plot.numpy(), S_f(t_plot).numpy(), 'k--', label='Fit des données S_f (cible)')
    ax.scatter(S_f.times, S_f.list_y, color='black', zorder=5, label='Points de données S_f')
    
    # Prédiction du modèle (via intégration)
    R_final_estimé = params.get("R_final_estimé", 0)
    nb_int_pts = 512 # Plus de points pour une belle courbe
    r_int_solide = torch.rand(nb_int_pts, 1) * R_final_estimé
    
    P_moy_pred_list = []
    for t_i in t_plot:
        t_int = t_i.repeat(nb_int_pts, 1)
        X_int = torch.cat([r_int_solide, t_int], dim=1)
        P_vals = model(X_int) # reload_model gère déjà la dé-normalisation
        P_moy_pred = 3.0 * torch.mean(P_vals * r_int_solide**2) / (R_final_estimé**3 + 1e-20)
        P_moy_pred_list.append(P_moy_pred.item())
        
    ax.plot(t_plot.numpy(), P_moy_pred_list, 'r-', label='Prédiction du modèle pour S_f')
    
    ax.set_title('Comparaison Données/Prédiction pour S_f', fontsize=16)
    ax.set_xlabel('Temps (s)', fontsize=12)
    ax.set_ylabel('Polarisation moyenne', fontsize=12)
    ax.grid(True, ls="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "Graphiques" / "S_f_comparison.png")
    plt.close(fig)

    # --- 3. Carte de chaleur de la Polarisation Ponctuelle P(r,t) ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    R_max = params.get("R_max", 400e-9)
    r_space = np.linspace(0, R_max, 150)
    t_space = np.linspace(0, params["def_t"], 150)
    
    grid_r, grid_t = np.meshgrid(r_space, t_space)
    X_flat = np.vstack([grid_r.ravel(), grid_t.ravel()]).T
    X_tensor = torch.from_numpy(X_flat).float()
    
    with torch.no_grad():
        P_pred = model(X_tensor).numpy().reshape(grid_r.shape)
        
    # On utilise pcolormesh pour une meilleure performance et précision
    c = ax.pcolormesh(grid_r * 1e9, grid_t, P_pred, cmap='jet', shading='gouraud')
    fig.colorbar(c, ax=ax, label='Polarisation P(r,t)')
    
    # Ajout d'une ligne verticale pour marquer le rayon estimé
    ax.axvline(x=R_final_estimé * 1e9, color='white', linestyle='--', linewidth=2, label=f'Rayon estimé R={R_final_estimé*1e9:.1f} nm')
    
    ax.set_title('Polarisation Ponctuelle P(r,t)', fontsize=16)
    ax.set_xlabel('Rayon (nm)', fontsize=12)
    ax.set_ylabel('Temps (s)', fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "Graphiques" / "P_rt_heatmap.png")
    plt.close(fig)


def cercle(path: Path, no_interaction: bool = False):
    """
    Génère la visualisation polaire interactive pour une expérience de type Axe 1.
    """
    path = Path(path)
    model = reload_model_Axe1(path=path)
    if isinstance(model, str):
        print(f"Erreur: {model}")
        return

    with open(path / "Data" / "params.json", "r") as f:
        params = json.load(f)

    R_final_estimé = params.get("R_final_estimé", 0)
    def_t = params.get("def_t", 20)
    P0_j = params.get("P0_j", 200)

    # Fonction pour calculer P(r,t) pour la colormap
    def get_P_at_time(r_coords, t):
        t_coords = np.full_like(r_coords, t)
        X_flat = np.vstack([r_coords.ravel(), t_coords.ravel()]).T
        X_tensor = torch.from_numpy(X_flat).float()
        with torch.no_grad():
            P_vals = model(X_tensor).numpy().reshape(r_coords.shape)
        return P_vals

    # Préparation de la grille polaire
    # On affiche un peu au-delà du rayon estimé pour voir le comportement dans le solvant
    r_space = np.linspace(0, R_final_estimé * 1.2, 100)
    theta_space = np.linspace(0, 2 * np.pi, 100)
    grid_theta, grid_r = np.meshgrid(theta_space, r_space)

    # Création de la figure
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 7))
    plt.subplots_adjust(bottom=0.25)
    
    # Données initiales pour t=0
    colors = get_P_at_time(grid_r, 0)
    
    c = ax.pcolormesh(grid_theta, grid_r * 1e9, colors, cmap='jet', shading='auto')
    cbar = fig.colorbar(c, ax=ax, pad=0.1)
    cbar.set_label('P(r,t)', size=12)
    c.set_clim(0, P0_j)

    ax.set_title(f'Polarisation pour R_estimé = {R_final_estimé*1e9:.1f} nm', pad=20)
    ax.set_xlabel('Rayon (nm)') # Note: l'axe r est radial
    
    # Slider pour le temps
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    time_slider = Slider(ax=ax_slider, label='Temps (s)', valmin=0.0, valmax=def_t, valinit=0.0)
    
    time_text = ax.text(0.5, 1.05, 't = 0.00 s', transform=ax.transAxes, ha='center')
    
    # Fonction de mise à jour appelée par le slider
    def update(val):
        t = time_slider.val
        colors = get_P_at_time(grid_r, t)
        c.set_array(colors.flatten())
        time_text.set_text(f't = {t:.2f} s')
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    
    if not no_interaction:
        plt.show()
    else:
        # Si non-interactif, on sauvegarde juste une image à t=def_t/2
        update(def_t / 2)
        fig.savefig(path / "Graphiques" / "cercle_snapshot.png")
        plt.close(fig)

# Tu peux garder cercle_for_frontend si tu en as besoin,
# il faudra l'adapter de la même manière que cercle.