# ==============================================================================
#           SINGLE SCRIPT FOR PINN-BASED SPIN DIFFUSION ANALYSIS
#               (Adapté pour le format de données .pkl)
# ==============================================================================

import json
import shutil
import torch
import pickle
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import copy
import math
import matplotlib.pyplot as plt

# ==============================================================================
# SECTION 1: OUTILS ET DÉFINITIONS FONDAMENTALES (anciennement tool.py)
# ==============================================================================

class Physics_informed_nn(nn.Module):
    """
    Définition de l'architecture du réseau de neurones.
    Le rayon R peut être un paramètre entraînable.
    """
    def __init__(self, nb_layer: int, hidden_layer: int, rayon_ini: float, coeff_normal: float, var_R: bool):
        super(Physics_informed_nn, self).__init__()
        self.coeff_normal = coeff_normal
        self.fc_int = nn.Linear(2, hidden_layer)
        self.fc = nn.ModuleList([nn.Linear(hidden_layer, hidden_layer) for _ in range(nb_layer)])
        self.fc_out = nn.Linear(hidden_layer, 1)
        if var_R:
            self.R = nn.Parameter(torch.tensor(float(rayon_ini), dtype=torch.float32, requires_grad=True))
        else:
            self.R = torch.tensor(rayon_ini, dtype=torch.float32)

    def forward(self, x):
        x = torch.tanh(self.fc_int(x))
        for fc in self.fc:
            x = torch.tanh(fc(x))
        x = torch.sigmoid(self.fc_out(x))
        return x

class Fick:
    """
    Classe représentant l'opérateur de l'équation de Fick.
    Calcule le résidu de l'EDP pour la fonction de coût.
    """
    def __init__(self, D, T, P_0):
        self.D = D
        self.T = T
        self.P_0 = P_0

    def __call__(self, P_func, X):
        X.requires_grad_(True)
        X_r = X[:, 0].view(-1, 1)
        
        # Dérivée de P par rapport aux entrées (r, t)
        dP_d = torch.autograd.grad(P_func, X, grad_outputs=torch.ones_like(P_func), create_graph=True)[0]
        dP_dr = dP_d[:, 0].view(-1, 1)
        dP_dt = dP_d[:, 1].view(-1, 1)

        # Dérivée seconde de P par rapport à r
        dP_drr = torch.autograd.grad(dP_dr, X, grad_outputs=torch.ones_like(dP_dr), create_graph=True)[0][:, 0].view(-1, 1)
        
        # Résidu de l'équation de Fick reformulée
        # Note: L'équation dans le rapport était P = (r/3)dG/dr + G. La Fick reformulée est appliquée à P.
        # Ici P est la polarisation, donc on utilise la Fick standard.
        # Le code original avait une Fick reformulée pour G, nous l'appliquons ici à P directement.
        # dP/dt = D * (d²P/dr² + (2/r)*dP/dr) - (P - P0)/T1
        # Multiplions par r pour éviter la singularité en r=0 :
        # r*dP/dt = D * (r*d²P/dr² + 2*dP/dr) - r*(P - P0)/T1
        
        # Le code original utilisait une formulation différente, nous la conservons pour la cohérence
        return X_r * dP_dt - self.D * (X_r * dP_drr + 2 * dP_dr) + X_r * ((P_func - self.P_0) / self.T)


def P_from_G(G, X):
    """Calcule la polarisation P à partir de la fonction auxiliaire G."""
    X_r = X[:, 0].view(-1, 1)
    dG_dr = torch.autograd.grad(G, X, grad_outputs=torch.ones_like(G), create_graph=True)[0][:, 0].view(-1, 1)
    return (X_r / 3) * dG_dr + G

def normalisation(R, D):
    """
    Normalise le rayon et le coefficient de diffusion pour la stabilité numérique.
    Retourne le rayon normalisé, D normalisé, et l'ordre de grandeur de R.
    """
    if R == 0:
        ordre_R = -7 # Valeur par défaut si R=0
    else:
        ordre_R = math.floor(math.log10(abs(R)))
    
    R_norm = R * 10**(-ordre_R)
    D_norm = D / (10**ordre_R)**2
    
    return R_norm, D_norm, ordre_R

class DataAugmentation:
    """
    MODIFICATION FONDAMENTALE: Normalise les données dès la création.
    La fonction __call__ retourne maintenant des valeurs normalisées (entre 0 et 1).
    """
    def __init__(self, data_df: pd.DataFrame, coeff_normal: float, mono=False):
        self.mono = mono
        self.coeff_normal = coeff_normal
        
        self.times = np.array(data_df["t"])
        # On normalise les données 'y' immédiatement
        self.list_y_raw = np.array(data_df["P_moy"])
        self.list_y_norm = self.list_y_raw / self.coeff_normal

        # Le fitting se fait sur les données normalisées
        self.tau, self.beta, self.C = 1.0, 1.0, max(self.list_y_norm) if len(self.list_y_norm) > 0 else 1.0
        best_params, self.min_loss = self._run_fit()

        if self.mono:
            self.tau, self.C = best_params
        else:
            self.tau, self.beta, self.C = best_params

    def __call__(self, t):
        t_np = t.detach().numpy() if isinstance(t, torch.Tensor) else t
        t_safe = np.where(t_np == 0, 1e-9, t_np)
        val = self.C * (1 - np.exp(-((t_safe / self.tau) ** self.beta)))
        val = np.where(t_np == 0, 0, val)
        return torch.tensor(val, dtype=torch.float32)

    def _cost(self, params):
        if self.mono:
            self.tau, self.C = params
        else:
            self.tau, self.beta, self.C = params
        y_pred = self.C * (1 - np.exp(-((self.times / self.tau) ** self.beta)))
        # Le coût est calculé sur les données normalisées
        return np.mean((y_pred - self.list_y_norm) ** 2)

    def _run_fit(self):
        # ... (le reste de la fonction _run_fit ne change pas)
        initial_params = [self.tau, self.beta, self.C]
        if self.mono:
            initial_params = [self.tau, self.C]
        result = minimize(
            self._cost,
            initial_params,
            method="L-BFGS-B",
            bounds=[(1e-6, None), (1e-6, None), (1e-6, None)] if not self.mono else [(1e-6, None), (1e-6, None)]
        )
        return result.x, result.fun

# ==============================================================================
# SECTION 2: FONCTIONS DE SAUVEGARDE ET VISUALISATION (anciennement save.py)
# ==============================================================================

def save_results(model, loss_history, params_pinns, params, path):
    """Sauvegarde le modèle, l'historique de la perte et les paramètres."""
    file_path = path / "Data"
    torch.save(model.state_dict(), file_path / "model.pth")

    with open(file_path / "loss.json", "w") as f:
        json.dump(loss_history, f)
    
    # On sauvegarde les deux dict de params pour la post-analyse
    with open(file_path / "params.json", "w") as f:
        json.dump(params, f, indent=4)
    with open(file_path / "params_PINNS.json", "w") as f:
        json.dump(params_pinns, f, indent=4)
    print(f"Résultats sauvegardés dans {file_path}")

def affichage(path: Path):
    """
    CORRIGÉ (Version finale): 
    - Lit le rayon depuis les paramètres sauvegardés pour garantir l'exactitude de l'affichage.
    - Assure la dé-normalisation correcte pour tous les graphiques.
    """
    print(f"Génération des graphiques pour les résultats dans : {path}")
    
    # --- Chargement des données et paramètres sauvegardés ---
    data_dir = path / "Data"
    graph_dir = path / "Graphiques"
    
    with open(data_dir / "loss.json", "r") as f:
        loss = json.load(f)
    with open(data_dir / "params.json", "r") as f:
        params = json.load(f)
    with open(data_dir / "params_PINNS.json", "r") as f:
        params_pinns = json.load(f)
    
    coeff_normal = params["P0_j"]
    
    with open(data_dir / "S_f.pkl", "rb") as f:
        S_f = pickle.load(f)
    with open(data_dir / "S_j.pkl", "rb") as f:
        S_j = pickle.load(f)

    # --- Reconstitution du modèle ---
    # Le rayon d'initialisation du modèle n'est pas critique, il sera écrasé
    model = Physics_informed_nn(
        nb_layer=params_pinns["nb_hidden_layer"],
        hidden_layer=params_pinns["nb_hidden_perceptron"],
        rayon_ini=0.1, # Valeur placeholder
        coeff_normal=coeff_normal,
        var_R=params_pinns["var_R"]
    )
    model.load_state_dict(torch.load(data_dir / "model.pth"))
    model.eval()
    
    # === CORRECTION DÉFINITIVE POUR L'AFFICHAGE DU RAYON ===
    # On lit le rayon qui a été VRAIMENT utilisé pendant l'entraînement depuis params.json
    R_final_m = params["rayon_initialisation"]
    R_final_norm, _, ordre_R = normalisation(R_final_m, params["D_f"])

    # Si le rayon était variable, on prend la dernière valeur de la perte
    if params_pinns["var_R"]:
        R_final_norm = loss[-1][-1]
        R_final_m = R_final_norm * (10**ordre_R)

    # --- Graphique 1: Évolution de la perte (et de R si variable) ---
    fig, axes = plt.subplots(1, 2 if params_pinns["var_R"] else 1, figsize=(20, 8), squeeze=False)
    ax1 = axes[0, 0]
    loss_names = ["Total Loss", "L_solid", "L_boundary", "L_initial", "L_fick"]
    for i, name in enumerate(loss_names):
        ax1.plot(loss[i], label=name)
    ax1.set_yscale('log')
    ax1.set_title('Evolution de la fonction de coût et de ses termes')
    ax1.set_xlabel('Itérations')
    ax1.set_ylabel('Coût (log)')
    ax1.legend()
    ax1.grid(True)

    if params_pinns["var_R"]:
        ax2 = axes[0, 1]
        R_vrai_m = params.get("R_vrai_m", 0)
        ax2.plot([r * (10**ordre_R) * 1e9 for r in loss[-1]], label=f'R prédit (final: {R_final_m*1e9:.1f} nm)')
        if R_vrai_m > 0:
            ax2.axhline(y=R_vrai_m * 1e9, color='r', linestyle='--', label=f'R vrai ({R_vrai_m*1e9:.1f} nm)')
        ax2.set_title('Evolution du rayon R prédit')
        ax2.set_xlabel('Itérations')
        ax2.set_ylabel('Rayon (nm)')
        ax2.legend()
        ax2.grid(True)
        
    fig.tight_layout()
    fig.savefig(graph_dir / "loss_evolution.png")
    plt.close(fig)

    # --- Graphique 2: Comparaison des courbes de polarisation moyenne ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    t_plot = torch.linspace(0, params["def_t"], 200).view(-1, 1)
    
    X_solid_boundary = torch.cat([torch.tensor(R_final_norm).repeat(t_plot.shape[0], 1), t_plot], dim=1)
    X_solid_boundary.requires_grad_(True)
    
    G_pred_norm = model(X_solid_boundary)
    P_pred_norm = P_from_G(G_pred_norm, X_solid_boundary)
    
    G_pred_denorm = G_pred_norm * coeff_normal
    P_pred_denorm = P_pred_norm * coeff_normal
    
    ax1.plot(t_plot.numpy(), S_f(t_plot).numpy() * coeff_normal, 'k--', label='S_f (données fittées)')
    ax1.plot(S_f.times, S_f.list_y_raw, 'ro', label='S_f (données brutes)')
    ax1.plot(t_plot.numpy(), G_pred_denorm.detach().numpy(), 'b-', label='Prédiction modèle G(R,t)')
    ax1.set_title('Polarisation moyenne du solide')
    ax1.set_xlabel('Temps (s)')
    ax1.set_ylabel('Polarisation')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t_plot.numpy(), S_j(t_plot).numpy() * coeff_normal, 'k--', label='S_j (données fittées)')
    ax2.plot(S_j.times, S_j.list_y_raw, 'ro', label='S_j (données brutes)')
    ax2.plot(t_plot.numpy(), P_pred_denorm.detach().numpy(), 'b-', label='Prédiction modèle P(R,t)')
    ax2.set_title('Polarisation moyenne du solvant (bord)')
    ax2.set_xlabel('Temps (s)')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(graph_dir / "mean_polarization_fit.png")
    plt.close(fig)

    # --- Graphique 3 et sauvegarde des .npy ---
    r_range = torch.linspace(0, R_final_m, 100)
    t_range = torch.linspace(0, params["def_t"], 100)
    grid_r, grid_t = torch.meshgrid(r_range, t_range, indexing='ij')
    
    grid_r_norm = grid_r / (10**ordre_R)
    
    X_grid = torch.stack([grid_r_norm.flatten(), grid_t.flatten()], dim=1)
    X_grid.requires_grad_(True) 

    G_grid_norm = model(X_grid)
    P_grid_norm = P_from_G(G_grid_norm, X_grid)
    P_grid_denorm = P_grid_norm * coeff_normal
    
    P_colormap = P_grid_denorm.detach().numpy().reshape(grid_r.shape)
    
    np.save(data_dir / "P.npy", P_colormap)
    np.save(data_dir / "(r, t).npy", (grid_r.numpy(), grid_t.numpy()))

    plt.figure(figsize=(10, 8))
    plt.contourf(grid_r.numpy() * 1e9, grid_t.numpy(), P_colormap, 50, cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Polarisation P(r,t)')
    plt.xlabel('Rayon r (nm)')
    plt.ylabel('Temps t (s)')
    plt.title(f"Polarisation ponctuelle prédite - R final = {R_final_m * 1e9:.1f} nm")
    plt.savefig(graph_dir / "P_r_t_colormap.png")
    plt.close()
# ==============================================================================
# SECTION 3: MOTEUR D'ENTRAÎNEMENT (anciennement PINN.py)
# ==============================================================================

def cost_function(model, F_f, S_f, S_j, def_t):
    """
    CORRECTION FINALE ET CIBLÉE:
    Modifie l'échantillonnage des points de collocation pour la perte de Fick
    afin d'éviter la solution triviale (plate).
    """
    R = model.R 
    nb_pts_t = 100 
    nb_pts_r = 100 

    t_points = torch.linspace(0, def_t, nb_pts_t).view(-1, 1)

    # --- Pertes aux bords (inchangées) ---
    X_solid_boundary = torch.cat([R.repeat(nb_pts_t, 1), t_points], dim=1)
    X_solid_boundary.requires_grad_(True)
    G_pred_norm = model(X_solid_boundary)
    P_pred_norm = P_from_G(G_pred_norm, X_solid_boundary)
    L_solid = torch.mean(torch.square(G_pred_norm - S_f(t_points)))
    L_boundary = torch.mean(torch.square(P_pred_norm - S_j(t_points)))

    # --- Perte initiale (inchangée) ---
    r_points_ini = torch.linspace(0, R.item(), nb_pts_r).view(-1, 1)
    t_points_ini = torch.zeros_like(r_points_ini)
    X_initial = torch.cat([r_points_ini, t_points_ini], dim=1)
    X_initial.requires_grad_(True)
    G_initial_norm = model(X_initial)
    P_initial_norm = P_from_G(G_initial_norm, X_initial)
    L_initial = torch.mean(torch.square(P_initial_norm))

    # --- Perte de Fick avec ÉCHANTILLONNAGE CORRIGÉ ---
    # On génère des points aléatoires, pas une grille fixe, pour mieux explorer l'espace.
    # On utilise une transformation cubique pour avoir moins de points près de r=0.
    
    # 1. Échantillonner des nombres aléatoires entre 0 et 1
    r_samples_uniform = torch.rand(nb_pts_r, 1)
    
    # 2. Appliquer une transformation cubique et mettre à l'échelle de R
    r_fick = (r_samples_uniform ** (1/3)) * R.item()
    
    # Échantillonner le temps de manière aléatoire aussi
    t_fick = torch.rand(nb_pts_t, 1) * def_t
    
    # Créer la grille de collocation
    grid_r_f, grid_t_f = torch.meshgrid(r_fick.squeeze(), t_fick.squeeze(), indexing="ij")
    X_fick = torch.stack([grid_r_f.flatten(), grid_t_f.flatten()], dim=1)
    X_fick.requires_grad_(True)
    
    G_fick_norm = model(X_fick)
    P_fick_norm = P_from_G(G_fick_norm, X_fick)
    
    residu_fick = F_f(P_fick_norm, X_fick)
    L_fick = torch.mean(torch.square(residu_fick))

    # On garde la pondération qui stabilise la convergence vers les données.
    lambda_data = 10.0
    lambda_fick = 1.0 # On peut même revenir à 1.0 car l'échantillonnage est plus intelligent
    
    total_loss = (lambda_data * (L_solid + L_boundary)) + L_initial + (lambda_fick * L_fick)

    loss_components = [total_loss.item(), L_solid.item(), L_boundary.item(), L_initial.item(), L_fick.item()]
    
    return total_loss, loss_components

def run(params_pinns: dict, params: dict, S_f: DataAugmentation, S_j: DataAugmentation, output_path: Path):
    """
    MODIFIÉ: Fonction principale d'entraînement.
    Prend S_f et S_j directement en argument.
    """
    # --- Initialisation ---
    torch.manual_seed(1234)
    var_R = params_pinns["var_R"]
    
    # Normalisation des paramètres physiques
    rayon_norm, D_f_norm, ordre_R = normalisation(params["rayon_initialisation"], params["D_f"])
    params["ordre_R"] = ordre_R # Sauvegarde pour post-traitement

    model = Physics_informed_nn(
        nb_layer=params_pinns["nb_hidden_layer"],
        hidden_layer=params_pinns["nb_hidden_perceptron"],
        rayon_ini=rayon_norm,
        coeff_normal=params["P0_j"], # Le plus grand P0 pour la normalisation de G
        var_R=var_R,
    )

    # P0_f est normalisé par P0_j car la sortie du réseau est entre 0 et 1 (sigmoid)
    P0_f_norm = params["P0_f"] / params["P0_j"]
    F_f = Fick(D_f_norm, params["T_1"], P0_f_norm)

    # --- Configuration de l'optimiseur ---
    # Stratégie: Entraîner d'abord le réseau avec R fixe, puis activer l'apprentissage de R
    if var_R:
        params_without_R = [p for name, p in model.named_parameters() if name != "R"]
        optimizer = optim.Adam([
            {'params': params_without_R, 'lr': params_pinns['lr']},
            {'params': model.R, 'lr': 0.0} # lr pour R est nul au début
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=params_pinns['lr'])
    
    loss_history = [[] for _ in range(5)] # L_tot, L_sol, L_bound, L_ini, L_fick
    if var_R:
        loss_history.append([]) # Pour stocker l'évolution de R

    best_loss = float('inf')
    model_opti = None

    # --- Boucle d'entraînement ---
    for it in tqdm(range(params_pinns["epoch"]), desc="Entraînement"):
        # Activer l'apprentissage de R après un certain nombre d'itérations
        if var_R and it == 3000:
            optimizer.param_groups[1]['lr'] = params_pinns['lr_R']
            print(f"\nItération {it}: Activation de l'apprentissage du rayon R avec lr={params_pinns['lr_R']}")

        optimizer.zero_grad()
        
        loss, loss_comp = cost_function(model, F_f, S_f, S_j, params["def_t"])
        
        loss.backward()
        optimizer.step()
        
        # Sauvegarde de l'historique
        for i in range(5):
            loss_history[i].append(loss_comp[i])
        if var_R:
            loss_history[5].append(model.R.item())

        # Sauvegarde du meilleur modèle
        if loss_comp[0] < best_loss:
            best_loss = loss_comp[0]
            model_opti = copy.deepcopy(model)
    
    print(f"\nEntraînement terminé. Meilleure perte : {best_loss:.2e}")
    save_results(model_opti, loss_history, params_pinns, params, output_path)
    return model_opti

# ==============================================================================
# SECTION 4: SCRIPT PRINCIPAL D'EXÉCUTION
# ==============================================================================

if __name__ == "__main__":
    # === CONFIGURATION DE L'EXPÉRIENCE ===
    EXP_NAME_TO_RUN = "11_58_40_25" 
    CASE = "On" # "On" pour hyperpolarisation, "Off" pour dépolarisation.

    # --- Chemins et paramètres globaux ---
    # MODIFIÉ pour utiliser un chemin relatif simple
    current_dir = Path(__file__).resolve().parents[2]
    data_file = current_dir / "data_1" / "donnees.pkl"
    base_output = current_dir / "output" / "multiple_runs_ND_2111D"

    # Paramètres du réseau de neurones
    params_pinns = {
        "nb_hidden_layer": 2,
        "nb_hidden_perceptron": 32,
        "lr": 0.001,
        "lr_R": 0.0005,
        "epoch": 1000, # Augmenté pour de meilleurs résultats
        "var_R": False,   # Mis à True pour apprendre le rayon
    }

    # === DÉROULEMENT DU SCRIPT ===

    # 1. Chargement de toutes les données prétraitées
    print(f"Chargement du fichier de données : {data_file}")
    if not data_file.exists():
        raise FileNotFoundError(f"Le fichier de données '{data_file}' est introuvable. Assurez-vous qu'il est au bon endroit.")
    with open(data_file, "rb") as f:
        all_data = pickle.load(f)
    print("Données chargées.")

    # 2. Préparation de l'expérience sélectionnée
    print(f"\n=== Préparation de l'expérience : {EXP_NAME_TO_RUN}, Cas : {CASE} ===")

    if EXP_NAME_TO_RUN not in all_data:
        raise ValueError(f"L'expérience '{EXP_NAME_TO_RUN}' est introuvable dans les données.")

    exp_data = all_data[EXP_NAME_TO_RUN]
    solid_data_key = "Cris" + CASE
    solvent_data_key = "Juice" + CASE

    # Définition du chemin de sortie et nettoyage
    output_path = base_output / f"{EXP_NAME_TO_RUN}_{CASE}"
    print(f"Les résultats seront sauvegardés dans : {output_path}")

    if output_path.exists():
        print("Le dossier de sortie existe déjà. Il va être supprimé et recréé.")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "Data").mkdir()
    (output_path / "Graphiques").mkdir()

    # 3. Création des objets de données (courbes fittées)
    print("Création des objets S_f et S_j par augmentation de données...")
    solid_df = pd.DataFrame(exp_data[solid_data_key])
    solvent_df = pd.DataFrame(exp_data[solvent_data_key])

    # Utilisation de la classe DataAugmentation modifiée
    coeff_normal = exp_data[solvent_data_key]["P0_j"]

    # Ensuite, on le passe à DataAugmentation
    S_f = DataAugmentation(data_df=solid_df, coeff_normal=coeff_normal)
    S_j = DataAugmentation(data_df=solvent_df, coeff_normal=coeff_normal)
    with open(output_path / "Data" / "S_f.pkl", "wb") as f:
        pickle.dump(S_f, f)
    with open(output_path / "Data" / "S_j.pkl", "wb") as f:
        pickle.dump(S_j, f)
    # S_j_mono n'est pas utilisé par la suite, on peut le laisser de côté pour l'instant.

    print("... Objets S_f et S_j créés et sauvegardés.")

    # 4. Construction du dictionnaire de paramètres pour le PINN
    print("Construction des paramètres physiques pour l'entraînement...")
    params = {
        # Valeurs à vérifier/ajuster, pour l'instant je mets des placeholders
        "D_f": 5e-16, 
        "D_j": 2.85e-16,
        "T_1": exp_data["T_1"],
        "T_B": exp_data[solvent_data_key]["TB_j"],
        "P0_f": exp_data[solid_data_key]["P0_j"],
        "P0_j": exp_data[solvent_data_key]["P0_j"],
        "C_f": exp_data.get("C_f", 1.0), # .get pour la robustesse si la clé manque
        "C_j": exp_data.get("C_j", 1.0),
        "def_t": max(exp_data[solid_data_key]["t"]),
        "rayon_initialisation": exp_data["R_s"] * 1.5e-9, # Conversion nm -> m, *1.5 pour tester la convergence
        "name": f"{EXP_NAME_TO_RUN}_{CASE}",
        "R_vrai_m": exp_data["R_s"] * 1.0e-9, # Stocker la vraie valeur pour comparaison
    }

    # 5. Lancement de l'entraînement
    print(f"\nLancement de l'entraînement du PINN pour {params['name']}...")
    run(
        params_pinns=params_pinns,
        params=params,
        S_f=S_f,
        S_j=S_j,
        output_path=output_path,
    )
    print("... Entraînement terminé.")

    # 6. Post-traitement et affichage
    print(f"\nPost-traitement et affichage des résultats pour {params['name']}...")
    try:
        affichage(output_path)
        print("... Graphiques générés avec succès.")
    except Exception as e:
        print(f"[AVERTISSEMENT] Erreur lors de la génération des graphiques post-run : {e}")

    print("\n=== FIN DE L'EXPÉRIENCE ===")