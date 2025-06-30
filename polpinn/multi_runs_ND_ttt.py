# ==============================================================================
#           SCRIPT UNIQUE - VERSION FIDÈLE À L'ORIGINAL
#               Adapté pour le format de données .pkl
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
# SECTION 1: OUTILS ET DÉFINITIONS (Basé sur tool.py)
# ==============================================================================

class Physics_informed_nn(nn.Module):
    # FIDÈLE À L'ORIGINAL
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
    # FIDÈLE À L'ORIGINAL
    def __init__(self, D, T, P_0):
        self.D = D
        self.T = T
        self.P_0 = P_0

    def __call__(self, P_func, X):
        X.requires_grad_(True)
        X_r = X[:, 0].view(-1, 1)
        dP_d = torch.autograd.grad(P_func, X, grad_outputs=torch.ones_like(P_func), create_graph=True)[0]
        dP_dr = dP_d[:, 0].view(-1, 1)
        dP_dt = dP_d[:, 1].view(-1, 1)
        dP_drr = torch.autograd.grad(dP_dr, X, grad_outputs=torch.ones_like(dP_dr), create_graph=True)[0][:, 0].view(-1, 1)
        return X_r * dP_dt - self.D * (X_r * dP_drr + 2 * dP_dr) + X_r * ((P_func - self.P_0) / self.T)

def P_from_G(G, X):
    # FIDÈLE À L'ORIGINAL
    X.requires_grad_(True)
    X_r = X[:, 0].view(-1, 1)
    dG_dr = torch.autograd.grad(G, X, grad_outputs=torch.ones_like(G), create_graph=True)[0][:, 0].view(-1, 1)
    return (X_r / 3) * dG_dr + G

def normalisation(R, D):
    # FIDÈLE À L'ORIGINAL
    if R == 0:
        ordre_R = -7 
    else:
        ordre_R = math.floor(math.log10(abs(R)))
    R_norm = R * 10**(-ordre_R)
    D_norm = D / (10**ordre_R)**2
    return R_norm, D_norm, ordre_R

class DataAugmentation:
    # MODIFIÉ pour accepter un DataFrame et normaliser les données
    def __init__(self, data_df: pd.DataFrame, coeff_normal: float, mono=False):
        self.mono = mono
        self.coeff_normal = coeff_normal
        self.times = np.array(data_df["t"])
        self.list_y_raw = np.array(data_df["P_moy"])
        self.list_y_norm = self.list_y_raw / self.coeff_normal
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
        return np.mean((y_pred - self.list_y_norm) ** 2)

    def _run_fit(self):
        initial_params = [self.tau, self.beta, self.C]
        if self.mono:
            initial_params = [self.tau, self.C]
        result = minimize(self._cost, initial_params, method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None), (1e-6, None)] if not self.mono else [(1e-6, None), (1e-6, None)])
        return result.x, result.fun

# ==============================================================================
# SECTION 2: MOTEUR D'ENTRAÎNEMENT (Basé sur PINN.py original)
# ==============================================================================

def cost_original(model, F_f, S_f, S_j, def_t):
    # === CORRIGÉ POUR LE GRAPHE DE CALCUL DE PYTORCH ===
    R = model.R
    nb_r = 30
    nb_t = int(def_t * 50) if def_t * 50 > 1 else 100
    
    # --- Perte L_solide ---
    # Pas de gradient nécessaire ici, donc pas de changement
    X_R_solide = R.repeat(nb_t, 1).view(-1, 1)
    X_t_solide = torch.linspace(0, def_t, nb_t).view(-1, 1)
    X_solide = torch.cat([X_R_solide, X_t_solide], dim=1)
    L_solide = torch.mean(torch.square(model(X_solide) - S_f(X_t_solide)))

    # --- Perte L_bord ---
    X_R_bord = R.repeat(nb_t, 1).view(-1, 1)
    X_t_bord = torch.linspace(0, def_t, nb_t).view(-1, 1)
    X_bord = torch.cat([X_R_bord, X_t_bord], dim=1)
    # === CORRECTION ===
    # On déclare le besoin de gradient AVANT de l'utiliser dans P_from_G
    X_bord.requires_grad_(True)
    L_bord = torch.mean(torch.square(P_from_G(model(X_bord), X_bord) - S_j(X_t_bord)))

    # --- Perte L_ini ---
    X_r_ini = torch.linspace(0, R.item(), nb_t).view(-1, 1)
    X_t_0 = torch.zeros((nb_t, 1)).view(-1, 1)
    X_ini = torch.cat([X_r_ini, X_t_0], dim=1)
    # === CORRECTION ===
    X_ini.requires_grad_(True)
    L_ini = torch.mean(torch.square(P_from_G(model(X_ini), X_ini)))

    # --- Perte L_fick ---
    X_r_f = torch.linspace(0, R.item(), nb_r).view(-1, 1)
    X_t_f = torch.linspace(0, def_t, nb_r).view(-1, 1)
    grid_r_f, grid_t_f = torch.meshgrid(X_r_f.squeeze(), X_t_f.squeeze(), indexing="ij")
    X_f = torch.stack([grid_r_f.flatten(), grid_t_f.flatten()], dim=1)
    # === CORRECTION ===
    X_f.requires_grad_(True)
    L_fick_f = torch.mean(torch.square(F_f(P_from_G(model(X_f), X_f), X_f)))

    # --- Pondération dynamique ---
    loss_sum = L_solide + L_bord + L_ini + L_fick_f
    if loss_sum.item() > 1e-12:
        gamma_solide = L_solide / loss_sum
        gamma_bord = L_bord / loss_sum
        gamma_ini = L_ini / loss_sum
        gamma_fick_f = L_fick_f / loss_sum
        total_loss = (gamma_solide * L_solide + gamma_bord * L_bord + gamma_ini * L_ini + gamma_fick_f * L_fick_f)
    else:
        total_loss = loss_sum

    loss_components = [loss_sum.item(), L_solide.item(), L_bord.item(), L_ini.item(), L_fick_f.item()]
    return total_loss, loss_components

def run_original(params_pinns: dict, params: dict, S_f: DataAugmentation, S_j: DataAugmentation, output_path: Path):
    # === FIDÈLE À LA FONCTION 'run' ORIGINALE ===
    torch.manual_seed(1234)
    var_R = params_pinns["var_R"]
    
    rayon_ini_norm, D_f_norm, ordre_R = normalisation(params["rayon_initialisation"], params["D_f"])
    params["ordre_R"] = ordre_R

    model = Physics_informed_nn(
        nb_layer=params_pinns["nb_hidden_layer"],
        hidden_layer=params_pinns["nb_hidden_perceptron"],
        rayon_ini=rayon_ini_norm,
        coeff_normal=params["P0_j"],
        var_R=var_R,
    )
    
    P0_f_norm = params["P0_f"] / params["P0_j"]
    F_f = Fick(D_f_norm, params["T_1"], P0_f_norm)

    if var_R:
        params_without_R = [param for name, param in model.named_parameters() if name != "R"]
        optimizer = optim.Adam([{"params": params_without_R}, {"params": model.R, "lr": 0}], lr=params_pinns["lr"])
        loss = [[] for _ in range(6)]
    else:
        optimizer = optim.Rprop(model.parameters())
        loss = [[] for _ in range(5)]

    model_opti = copy.deepcopy(model)
    min_loss_val = float('inf')

    for it in tqdm(range(params_pinns["epoch"]), desc="Entraînement"):
        if it == 3000 and var_R:
            params_without_R = [param for name, param in model.named_parameters() if name != "R"]
            optimizer = optim.Adam(
                [{"params": params_without_R, "lr": params_pinns["lr"]},
                 {"params": model.R, "lr": params_pinns["lr_R"]}],
                lr=params_pinns["lr"],
            )
        elif it == 3000 and not var_R:
            optimizer = optim.Adam(model.parameters(), lr=params_pinns['lr'])

        optimizer.zero_grad()
        L, L_total_list = cost_original(model, F_f, S_f, S_j, params["def_t"])
        L.backward()
        optimizer.step()
        
        for i in range(len(L_total_list)):
            loss[i].append(L_total_list[i])
        if var_R:
            loss[-1].append(model.R.item())

        if L_total_list[0] < min_loss_val:
            min_loss_val = L_total_list[0]
            model_opti = copy.deepcopy(model)

    print(f"\nEntraînement terminé. Meilleure perte (sum): {min_loss_val:.2e}")
    return model_opti, loss

# ==============================================================================
# SECTION 3: SAUVEGARDE ET VISUALISATION (Adapté de save.py)
# ==============================================================================
def save_results(model, loss_history, params_pinns, params, path):
    file_path = path / "Data"
    torch.save(model.state_dict(), file_path / "model.pth")
    with open(file_path / "loss.json", "w") as f:
        json.dump(loss_history, f)
    with open(file_path / "params.json", "w") as f:
        json.dump(params, f, indent=4)
    with open(file_path / "params_PINNS.json", "w") as f:
        json.dump(params_pinns, f, indent=4)
    print(f"Résultats sauvegardés dans {file_path}")

def affichage(path: Path):
    print(f"Génération des graphiques pour les résultats dans : {path}")
    
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

    rayon_initial_m = params["rayon_initialisation"]
    R_norm, _, ordre_R = normalisation(rayon_initial_m, params["D_f"])

    model = Physics_informed_nn(
        nb_layer=params_pinns["nb_hidden_layer"],
        hidden_layer=params_pinns["nb_hidden_perceptron"],
        rayon_ini=R_norm, 
        coeff_normal=coeff_normal,
        var_R=params_pinns["var_R"]
    )
    model.load_state_dict(torch.load(data_dir / "model.pth"))
    model.eval()
    
    if params_pinns["var_R"]:
        R_final_norm = loss[-1][-1]
    else:
        R_final_norm = model.R.item()
    R_final_m = R_final_norm * (10**ordre_R)
    
    # --- Graphique 1: Évolution de la perte (et de R si variable) ---
    fig, axes = plt.subplots(1, 2 if params_pinns["var_R"] else 1, figsize=(20, 8), squeeze=False)
    ax1 = axes[0, 0]
    loss_names = ["Total Sum", "L_solid", "L_boundary", "L_initial", "L_fick"]
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
    
    # === CORRECTION ICI ===
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

    # === CORRECTION ICI ===
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
# SECTION 4: SCRIPT PRINCIPAL D'EXÉCUTION
# ==============================================================================

if __name__ == "__main__":
    # --- CONFIGURATION ---
    EXP_NAME_TO_RUN = "11_58_40_75" 
    CASE = "On"
    params_pinns = {
        "nb_hidden_layer": 2,
        "nb_hidden_perceptron": 32,
        "lr": 0.001,
        "lr_R": 0.0005,
        "epoch": 1000, # Un nombre d'itérations plus élevé pour converger
        "var_R": False,
    }

    # --- CHEMINS ---
    code_dir = Path(__file__).resolve().parents[2]
    data_file = code_dir / "data_1" / "donnees.pkl"
    base_output = code_dir / "output" / "runs_fidele_original_1"

    # --- CHARGEMENT DES DONNÉES ---
    with open(data_file, "rb") as f:
        all_data = pickle.load(f)

    # --- PRÉPARATION DE L'EXPÉRIENCE ---
    exp_data = all_data[EXP_NAME_TO_RUN]
    solid_data_key = "Cris" + CASE
    solvent_data_key = "Juice" + CASE
    output_path = base_output / f"{EXP_NAME_TO_RUN}_{CASE}"
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "Data").mkdir()
    (output_path / "Graphiques").mkdir()

    # --- CALCUL DES PARAMÈTRES PHYSIQUES (crucial) ---
    C_ref, D_ref_nm2_s = 60.0, 500.0
    D_ref_m2_s = D_ref_nm2_s * 1e-18
    C_f = exp_data.get("C_f", C_ref)
    C_j = exp_data.get("C_j", C_ref)
    D_f_calculated = D_ref_m2_s * ((C_f / C_ref) ** (1/3))
    
    params = {
        "D_f": D_f_calculated,
        "T_1": exp_data["T_1"],
        "P0_f": exp_data[solid_data_key]["P0_j"],
        "P0_j": exp_data[solvent_data_key]["P0_j"],
        "rayon_initialisation": exp_data["R_s"] * 1.0e-9,
        "def_t": max(exp_data[solid_data_key]["t"]),
        "name": f"{EXP_NAME_TO_RUN}_{CASE}",
        "R_vrai_m": exp_data["R_s"] * 1.0e-9,
    }
    
    # --- PRÉPARATION DES DONNÉES NORMALISÉES ---
    coeff_normal = params["P0_j"]
    solid_df = pd.DataFrame(exp_data[solid_data_key])
    solvent_df = pd.DataFrame(exp_data[solvent_data_key])
    S_f = DataAugmentation(data_df=solid_df, coeff_normal=coeff_normal)
    S_j = DataAugmentation(data_df=solvent_df, coeff_normal=coeff_normal)

    # --- LANCEMENT ---
    model_final, loss_history = run_original(params_pinns, params, S_f, S_j, output_path)

    # --- SAUVEGARDE ---
    save_results(model_final, loss_history, params_pinns, params, output_path)
    with open(output_path / "Data" / "S_f.pkl", "wb") as f:
        pickle.dump(S_f, f)
    with open(output_path / "Data" / "S_j.pkl", "wb") as f:
        pickle.dump(S_j, f)

    # --- AFFICHAGE ---
    affichage(output_path)

    print("\n=== FIN DE L'EXPÉRIENCE ===")