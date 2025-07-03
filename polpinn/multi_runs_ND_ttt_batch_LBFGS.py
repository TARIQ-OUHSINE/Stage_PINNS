# ==============================================================================
#           SCRIPT UNIQUE - VERSION AVEC MINI-BATCHS
#               Adapté pour le format de données .pkl
# ==============================================================================

import sys
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
import argparse ### MODIFIÉ ###: Ajout de la bibliothèque pour les arguments

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
# SECTION 2: MOTEUR D'ENTRAÎNEMENT PAR MINI-BATCHS ET L-BFGS (AUCUN CHANGEMENT ICI)
# ==============================================================================

def cost_original_batch(model, F_f, S_f, S_j, X_fick_batch, X_data_batch):
    R = model.R
    X_fick_batch.requires_grad_(True)
    L_fick_f = torch.mean(torch.square(F_f(P_from_G(model(X_fick_batch), X_fick_batch), X_fick_batch)))
    X_data_batch.requires_grad_(True)
    t_vals = X_data_batch[:, 1]
    X_boundary_batch = X_data_batch[t_vals > 0]
    t_boundary_batch = X_boundary_batch[:, 1].view(-1, 1)
    X_ini_batch = X_data_batch[t_vals == 0]
    L_solide = torch.mean(torch.square(model(X_boundary_batch) - S_f(t_boundary_batch)))
    L_bord = torch.mean(torch.square(P_from_G(model(X_boundary_batch), X_boundary_batch) - S_j(t_boundary_batch)))
    if X_ini_batch.shape[0] > 0:
        L_ini = torch.mean(torch.square(P_from_G(model(X_ini_batch), X_ini_batch)))
    else:
        L_ini = torch.tensor(0.0, device=X_fick_batch.device)
    loss_sum = L_solide + L_bord + L_ini + L_fick_f
    if loss_sum.item() > 1e-12:
        gamma_solide = L_solide / loss_sum; gamma_bord = L_bord / loss_sum
        gamma_ini = L_ini / loss_sum; gamma_fick_f = L_fick_f / loss_sum
        total_loss = (gamma_solide * L_solide + gamma_bord * L_bord + gamma_ini * L_ini + gamma_fick_f * L_fick_f)
    else: total_loss = loss_sum
    loss_components = [loss_sum.item(), L_solide.item(), L_bord.item(), L_ini.item(), L_fick_f.item()]
    return total_loss, loss_components

def cost_full_batch(model, F_f, S_f, S_j, X_fick_total, X_data_total):
    R = model.R
    X_fick_total.requires_grad_(True)
    L_fick_f = torch.mean(torch.square(F_f(P_from_G(model(X_fick_total), X_fick_total), X_fick_total)))
    X_data_total.requires_grad_(True)
    t_vals = X_data_total[:, 1]
    X_boundary = X_data_total[t_vals > 0]; t_boundary = X_boundary[:, 1].view(-1, 1)
    X_ini = X_data_total[t_vals == 0]
    L_solide = torch.mean(torch.square(model(X_boundary) - S_f(t_boundary)))
    L_bord = torch.mean(torch.square(P_from_G(model(X_boundary), X_boundary) - S_j(t_boundary)))
    L_ini = torch.mean(torch.square(P_from_G(model(X_ini), X_ini)))
    loss_sum = L_solide + L_bord + L_ini + L_fick_f
    if loss_sum.item() > 1e-12:
        gamma_solide = L_solide / loss_sum; gamma_bord = L_bord / loss_sum
        gamma_ini = L_ini / loss_sum; gamma_fick_f = L_fick_f / loss_sum
        total_loss = (gamma_solide * L_solide + gamma_bord * L_bord + gamma_ini * L_ini + gamma_fick_f * L_fick_f)
    else: total_loss = loss_sum
    loss_components = [loss_sum.item(), L_solide.item(), L_bord.item(), L_ini.item(), L_fick_f.item()]
    return total_loss, loss_components

def run_original_batch(params_pinns: dict, params: dict, S_f: DataAugmentation, S_j: DataAugmentation, output_path: Path):
    # === PHASE DE SETUP ===
    torch.manual_seed(1234)
    var_R = params_pinns["var_R"]
    rayon_ini_norm, D_f_norm, ordre_R = normalisation(params["rayon_initialisation"], params["D_f"])
    params["ordre_R"] = ordre_R
    model = Physics_informed_nn(nb_layer=params_pinns["nb_hidden_layer"], hidden_layer=params_pinns["nb_hidden_perceptron"], rayon_ini=rayon_ini_norm, coeff_normal=params["P0_j"], var_R=var_R)
    P0_f_norm = params["P0_f"] / params["P0_j"]
    F_f = Fick(D_f_norm, params["T_1"], P0_f_norm)
    
    # ==============================================================================
    # --- MODIFICATIONS POUR UN TEST RAPIDE ---
    # ==============================================================================
    print("Création du DataSet de points de collocation (VERSION RAPIDE)...")
    def_t = params["def_t"]; R_item = rayon_ini_norm
    
    # 1. Réduction du nombre de points
    nb_r_total = 100
    nb_t_total = 100  # sqrt(10000)
    
    # 2. Réduction du nombre de points de données
    nb_t_data = 100 

    # --- Création du DataSet avec les nouvelles tailles ---
    X_r_f_total = torch.linspace(0, R_item, nb_r_total).view(-1, 1)
    X_t_f_total = torch.linspace(0, def_t, nb_t_total).view(-1, 1)
    grid_r_f, grid_t_f = torch.meshgrid(X_r_f_total.squeeze(), X_t_f_total.squeeze(), indexing="ij")
    X_fick_total = torch.stack([grid_r_f.flatten(), grid_t_f.flatten()], dim=1)   # Total = 10000 points
    
    X_R_data_total = torch.full((nb_t_data, 1), R_item); X_t_data_total = torch.linspace(0, def_t, nb_t_data).view(-1, 1)
    X_boundary_total = torch.cat([X_R_data_total, X_t_data_total], dim=1)
    X_r_ini_total = torch.linspace(0, R_item, nb_t_data).view(-1, 1); X_t_ini_total = torch.zeros((nb_t_data, 1))
    X_ini_total = torch.cat([X_r_ini_total, X_t_ini_total], dim=1)
    X_data_total = torch.cat([X_boundary_total, X_ini_total], dim=0)   # Total = 200 points
    
    print(f"DataSet créé: {X_fick_total.shape[0]} points de physique, {X_data_total.shape[0]} points de données.")

    loss = [[] for _ in range(5)]
    if var_R: loss.append([])
    model_opti = copy.deepcopy(model)
    min_loss_val = float('inf')

    # --- PHASE 1: ADAM - NOUVELLE LOGIQUE DE MINI-BATCHING COMPLET ---
    print("\n--- Phase 1: Adam Optimizer (Mini-Batching Complet) ---")
    optimizer = optim.Adam(model.parameters(), lr=params_pinns['lr'])
    epochs_phase1 = 10 # 10 époques
    
    # 3. Définition des tailles de batch
    fick_batch_size = 1000
    data_batch_size = 10
    
    # Création des DataLoaders pour gérer les batchs facilement
    fick_dataset = torch.utils.data.TensorDataset(X_fick_total)
    fick_loader = torch.utils.data.DataLoader(fick_dataset, batch_size=fick_batch_size, shuffle=True)
    data_dataset = torch.utils.data.TensorDataset(X_data_total)
    data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=data_batch_size, shuffle=True)

    for epoch in range(epochs_phase1):
        epoch_loss_sum = 0.0
        # On itère sur les deux loaders en parallèle
        # zip s'arrêtera quand le plus petit loader sera épuisé (celui des données)
        for (fick_batch,), (data_batch,) in tqdm(zip(fick_loader, data_loader), desc=f"Epoch {epoch+1}/{epochs_phase1} (Adam)", file=sys.stdout):
            optimizer.zero_grad()
            L, L_total_list = cost_original_batch(model, F_f, S_f, S_j, fick_batch, data_batch)
            L.backward(); optimizer.step()
            epoch_loss_sum += L_total_list[0]

        # Logging à la fin de chaque époque complète
        avg_epoch_loss = epoch_loss_sum / len(data_loader)
        if avg_epoch_loss < min_loss_val: 
            min_loss_val = avg_epoch_loss
            model_opti = copy.deepcopy(model)
        
        # On log les composantes du dernier batch pour le graphique
        for i in range(len(L_total_list)): loss[i].append(L_total_list[i])
        if var_R: loss[-1].append(model.R.item())

    ### On repart du MEILLEUR modèle (inchangé) ###
    print(f"\nFin de la phase Adam. Meilleure perte trouvée : {min_loss_val:.2e}")
    print("Chargement du meilleur modèle pour L-BFGS...")
    model.load_state_dict(model_opti.state_dict())
    
    # --- PHASE 2: L-BFGS - TEST RAPIDE ---
    print("\n--- Phase 2: L-BFGS Optimizer (Test Rapide) ---")
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn="strong_wolfe")
    last_closure_metrics = {}
    epochs_phase2 = 10 # 10 époques pour le test

    for epoch in tqdm(range(epochs_phase2), desc="Phase 2 (L-BFGS)", file=sys.stdout):
        def closure():
            optimizer_lbfgs.zero_grad()
            L, L_total_list = cost_full_batch(model, F_f, S_f, S_j, X_fick_total, X_data_total)
            L.backward()
            nonlocal last_closure_metrics
            last_closure_metrics['components'] = L_total_list
            if var_R: last_closure_metrics['R'] = model.R.item()
            return L
        
        optimizer_lbfgs.step(closure)
        final_epoch_components = last_closure_metrics.get('components', [0]*5)
        for i in range(len(final_epoch_components)): loss[i].append(final_epoch_components[i])
        if var_R: loss[-1].append(last_closure_metrics.get('R', model.R.item()))
        current_loss = final_epoch_components[0]
        if current_loss < min_loss_val:
            min_loss_val = current_loss
            model_opti = copy.deepcopy(model)

    print(f"\nEntraînement terminé. Meilleure perte finale (sum): {min_loss_val:.2e}")
    return model_opti, loss

# ==============================================================================
# SECTION 3: SAUVEGARDE ET VISUALISATION (AUCUN CHANGEMENT ICI)
# ==============================================================================
def save_results(model, loss_history, params_pinns, params, path):
    file_path = path / "Data"
    torch.save(model.state_dict(), file_path / "model.pth")
    with open(file_path / "loss.json", "w") as f: json.dump(loss_history, f)
    with open(file_path / "params.json", "w") as f: json.dump(params, f, indent=4)
    with open(file_path / "params_PINNS.json", "w") as f: json.dump(params_pinns, f, indent=4)
    print(f"Résultats sauvegardés dans {file_path}")

def affichage(path: Path):
    print(f"Génération des graphiques pour les résultats dans : {path}")
    data_dir = path / "Data"; graph_dir = path / "Graphiques"
    with open(data_dir / "loss.json", "r") as f: loss = json.load(f)
    with open(data_dir / "params.json", "r") as f: params = json.load(f)
    with open(data_dir / "params_PINNS.json", "r") as f: params_pinns = json.load(f)
    coeff_normal = params["P0_j"]
    with open(data_dir / "S_f.pkl", "rb") as f: S_f = pickle.load(f)
    with open(data_dir / "S_j.pkl", "rb") as f: S_j = pickle.load(f)
    rayon_initial_m = params["rayon_initialisation"]
    R_norm, _, ordre_R = normalisation(rayon_initial_m, params["D_f"])
    model = Physics_informed_nn(nb_layer=params_pinns["nb_hidden_layer"], hidden_layer=params_pinns["nb_hidden_perceptron"], rayon_ini=R_norm, coeff_normal=coeff_normal, var_R=params_pinns["var_R"])
    model.load_state_dict(torch.load(data_dir / "model.pth")); model.eval()
    if params_pinns["var_R"]: R_final_norm = loss[-1][-1]
    else: R_final_norm = model.R.item()
    R_final_m = R_final_norm * (10**ordre_R)
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 8))
    loss_names = ["Total Sum", "L_solid", "L_boundary", "L_initial", "L_fick"]
    for i, name in enumerate(loss_names): ax1.plot(loss[i], label=name)
    ax1.set_yscale('log'); ax1.set_title('Evolution de la fonction de coût et de ses termes'); ax1.set_xlabel('Itérations (x10)'); ax1.set_ylabel('Coût (log)'); ax1.legend(); ax1.grid(True)
    fig.tight_layout(); fig.savefig(graph_dir / "loss_evolution.png"); plt.close(fig)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    t_plot = torch.linspace(0, params["def_t"], 200).view(-1, 1)
    X_solid_boundary = torch.cat([torch.tensor(R_final_norm).repeat(t_plot.shape[0], 1), t_plot], dim=1); X_solid_boundary.requires_grad_(True)
    G_pred_norm = model(X_solid_boundary); P_pred_norm = P_from_G(G_pred_norm, X_solid_boundary)
    G_pred_denorm = G_pred_norm * coeff_normal; P_pred_denorm = P_pred_norm * coeff_normal
    ax1.plot(t_plot.numpy(), S_f(t_plot).numpy() * coeff_normal, 'k--', label='S_f (données fittées)'); ax1.plot(S_f.times, S_f.list_y_raw, 'ro', label='S_f (données brutes)'); ax1.plot(t_plot.numpy(), G_pred_denorm.detach().numpy(), 'b-', label='Prédiction modèle G(R,t)')
    ax1.set_title('Polarisation moyenne du solide'); ax1.set_xlabel('Temps (s)'); ax1.set_ylabel('Polarisation'); ax1.legend(); ax1.grid(True)
    ax2.plot(t_plot.numpy(), S_j(t_plot).numpy() * coeff_normal, 'k--', label='S_j (données fittées)'); ax2.plot(S_j.times, S_j.list_y_raw, 'ro', label='S_j (données brutes)'); ax2.plot(t_plot.numpy(), P_pred_denorm.detach().numpy(), 'b-', label='Prédiction modèle P(R,t)')
    ax2.set_title('Polarisation moyenne du solvant (bord)'); ax2.set_xlabel('Temps (s)'); ax2.legend(); ax2.grid(True)
    fig.tight_layout(); fig.savefig(graph_dir / "mean_polarization_fit.png"); plt.close(fig)
    r_range = torch.linspace(0, R_final_m, 100); t_range = torch.linspace(0, params["def_t"], 100)
    grid_r, grid_t = torch.meshgrid(r_range, t_range, indexing='ij')
    grid_r_norm = grid_r / (10**ordre_R)
    X_grid = torch.stack([grid_r_norm.flatten(), grid_t.flatten()], dim=1); X_grid.requires_grad_(True)
    G_grid_norm = model(X_grid); P_grid_norm = P_from_G(G_grid_norm, X_grid); P_grid_denorm = P_grid_norm * coeff_normal
    P_colormap = P_grid_denorm.detach().numpy().reshape(grid_r.shape)
    np.save(data_dir / "P.npy", P_colormap); np.save(data_dir / "(r, t).npy", (grid_r.numpy(), grid_t.numpy()))
    plt.figure(figsize=(10, 8))
    plt.contourf(grid_r.numpy() * 1e9, grid_t.numpy(), P_colormap, 50, cmap='jet')
    cbar = plt.colorbar(); cbar.set_label('Polarisation P(r,t)')
    plt.xlabel('Rayon r (nm)'); plt.ylabel('Temps t (s)'); plt.title(f"Polarisation ponctuelle prédite - R final = {R_final_m * 1e9:.1f} nm")
    plt.savefig(graph_dir / "P_r_t_colormap.png"); plt.close()

# ==============================================================================
# SECTION 4: SCRIPT PRINCIPAL D'EXÉCUTION
# ==============================================================================

if __name__ == "__main__":
    ### MODIFIÉ ###: Mise en place du parser d'arguments
    parser = argparse.ArgumentParser(description="Lancement d'une simulation PINN pour un cas spécifique.")
    parser.add_argument('--data_file', type=str, required=True, help="Chemin vers le fichier de données principal (donnees.pkl)")
    parser.add_argument('--output_dir', type=str, required=True, help="Chemin vers le dossier racine où les résultats seront sauvegardés.")
    parser.add_argument('--case_name', type=str, required=True, help="Nom du cas expérimental à traiter (ex: 11_58_40_25)")
    args = parser.parse_args()

    # --- CONFIGURATION (gardée pour les paramètres du PINN) ---
    CASE = "On"
    params_pinns = {
        "nb_hidden_layer": 2, "nb_hidden_perceptron": 32,
        "lr": 0.001, "lr_R": 0.0005,
        "epoch": 5000, "var_R": False,
        "batch_size": 100,
    }

    ### MODIFIÉ ###: Utilisation des arguments pour définir les chemins
    # Les chemins sont maintenant flexibles et donnés par l'utilisateur
    EXP_NAME_TO_RUN = args.case_name
    data_file = Path(args.data_file)
    base_output = Path(args.output_dir)

    # Le reste de la logique est inchangé, elle utilise les variables définies ci-dessus
    # --- CHARGEMENT ET PRÉPARATION ---
    print(f"Chargement des données depuis : {data_file}")
    print(f"Cas à traiter : {EXP_NAME_TO_RUN}")
    print(f"Les résultats seront sauvegardés dans : {base_output}")

    with open(data_file, "rb") as f: all_data = pickle.load(f)
    exp_data = all_data[EXP_NAME_TO_RUN]
    solid_data_key, solvent_data_key = "Cris" + CASE, "Juice" + CASE
    output_path = base_output / f"{EXP_NAME_TO_RUN}_{CASE}"
    if output_path.exists(): shutil.rmtree(output_path)
    (output_path / "Data").mkdir(parents=True, exist_ok=True)
    (output_path / "Graphiques").mkdir(parents=True, exist_ok=True)

    # --- CALCUL DES PARAMÈTRES PHYSIQUES ---
    C_ref, D_ref_nm2_s = 60.0, 500.0
    D_ref_m2_s = D_ref_nm2_s * 1e-18
    C_f, C_j = exp_data.get("C_f", C_ref), exp_data.get("C_j", C_ref)
    D_f_calculated = D_ref_m2_s * ((C_f / C_ref) ** (1/3))
    params = {
        "D_f": D_f_calculated, "T_1": exp_data["T_1"],
        "P0_f": exp_data[solid_data_key]["P0_j"], "P0_j": exp_data[solvent_data_key]["P0_j"],
        "rayon_initialisation": exp_data["R_s"] * 1.0e-9,
        "def_t": max(exp_data[solid_data_key]["t"]),
        "name": f"{EXP_NAME_TO_RUN}_{CASE}", "R_vrai_m": exp_data["R_s"] * 1.0e-9,
    }
    
    # --- PRÉPARATION DES DONNÉES NORMALISÉES ---
    coeff_normal = params["P0_j"]
    solid_df, solvent_df = pd.DataFrame(exp_data[solid_data_key]), pd.DataFrame(exp_data[solvent_data_key])
    S_f = DataAugmentation(data_df=solid_df, coeff_normal=coeff_normal)
    S_j = DataAugmentation(data_df=solvent_df, coeff_normal=coeff_normal)

    # --- LANCEMENT ---
    model_final, loss_history = run_original_batch(params_pinns, params, S_f, S_j, output_path)

    # --- SAUVEGARDE ET AFFICHAGE ---
    save_results(model_final, loss_history, params_pinns, params, output_path)
    with open(output_path / "Data" / "S_f.pkl", "wb") as f: pickle.dump(S_f, f)
    with open(output_path / "Data" / "S_j.pkl", "wb") as f: pickle.dump(S_j, f)
    affichage(output_path)

    print("\n=== FIN DE L'EXPÉRIENCE ===")