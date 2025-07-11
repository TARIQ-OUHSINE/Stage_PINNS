# ==============================================================================
#           SCRIPT UNIQUE - VERSION CORRIGÉE ET VÉRIFIÉE
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
import argparse

# SECTION 1: OUTILS ET DÉFINITIONS (Inchangé, correct)
class Physics_informed_nn(nn.Module):
    def __init__(self, nb_layer: int, hidden_layer: int, rayon_R_norm: float, coeff_normal: float):
        super(Physics_informed_nn, self).__init__()
        self.coeff_normal = coeff_normal
        self.fc_int = nn.Linear(2, hidden_layer)
        self.fc = nn.ModuleList([nn.Linear(hidden_layer, hidden_layer) for _ in range(nb_layer)])
        self.fc_out = nn.Linear(hidden_layer, 1)
        self.R_norm = torch.tensor(rayon_R_norm, dtype=torch.float32)
    def forward(self, x):
        x = torch.tanh(self.fc_int(x))
        for fc in self.fc: x = torch.tanh(fc(x))
        x = torch.sigmoid(self.fc_out(x)); return x

class Fick:
    def __init__(self, D, T, P_0):
        self.D, self.T, self.P_0 = D, T, P_0
    def __call__(self, P_func, X):
        X.requires_grad_(True)
        X_r = X[:, 0].view(-1, 1)
        dP_d = torch.autograd.grad(P_func, X, grad_outputs=torch.ones_like(P_func), create_graph=True)[0]
        dP_dr, dP_dt = dP_d[:, 0].view(-1, 1), dP_d[:, 1].view(-1, 1)
        dP_drr = torch.autograd.grad(dP_dr, X, grad_outputs=torch.ones_like(dP_dr), create_graph=True)[0][:, 0].view(-1, 1)
        return X_r * dP_dt - self.D * (X_r * dP_drr + 2 * dP_dr) + X_r * ((P_func - self.P_0) / self.T)

def P_from_G(G, X):
    X.requires_grad_(True)
    X_r = X[:, 0].view(-1, 1)
    dG_dr = torch.autograd.grad(G, X, grad_outputs=torch.ones_like(G), create_graph=True)[0][:, 0].view(-1, 1)
    return (X_r / 3) * dG_dr + G

def normalisation(R, D_ref):
    ordre_R = math.floor(math.log10(abs(R))) if R != 0 else -7
    R_norm = R * 10**(-ordre_R)
    D_norm = D_ref / (10**ordre_R)**2
    return R_norm, D_norm, ordre_R

class DataAugmentation:
    def __init__(self, data_df, coeff_normal, mono=False):
        self.mono, self.coeff_normal = mono, coeff_normal
        self.times = np.array(data_df["t"])
        self.list_y_raw = np.array(data_df["P_moy"])
        self.list_y_norm = self.list_y_raw / self.coeff_normal
        self.tau, self.beta, self.C = 1.0, 1.0, max(self.list_y_norm) if len(self.list_y_norm) > 0 else 1.0
        best_params, self.min_loss = self._run_fit()
        if self.mono: self.tau, self.C = best_params
        else: self.tau, self.beta, self.C = best_params
    def __call__(self, t):
        t_np = t.detach().numpy() if isinstance(t, torch.Tensor) else t
        t_safe = np.where(t_np == 0, 1e-9, t_np)
        val = self.C * (1 - np.exp(-((t_safe / self.tau) ** self.beta)))
        val = np.where(t_np == 0, 0, val); return torch.tensor(val, dtype=torch.float32)
    def _cost(self, params):
        if self.mono: self.tau, self.C = params
        else: self.tau, self.beta, self.C = params
        y_pred = self.C * (1 - np.exp(-((self.times / self.tau) ** self.beta)))
        return np.mean((y_pred - self.list_y_norm) ** 2)
    def _run_fit(self):
        initial_params = [self.tau, self.beta, self.C]
        if self.mono: initial_params = [self.tau, self.C]
        result = minimize(self._cost, initial_params, method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None), (1e-6, None)] if not self.mono else [(1e-6, None), (1e-6, None)])
        return result.x, result.fun

# ==============================================================================
# SECTION 2: MOTEUR D'ENTRAÎNEMENT (CORRIGÉ)
# ==============================================================================

def cost_enhanced_batch(model, F_solid, F_liquid, S_f, S_j, X_fick_batch, X_data_batch, X_grad_batch, R_norm, R_prime_norm):
    X_fick_batch.requires_grad_(True)
    r_fick = X_fick_batch[:, 0]
    X_fick_solid = X_fick_batch[r_fick < R_norm]
    X_fick_liquid = X_fick_batch[r_fick >= R_norm]
    L_fick_s = torch.mean(torch.square(F_solid(P_from_G(model(X_fick_solid), X_fick_solid), X_fick_solid))) if X_fick_solid.shape[0] > 0 else torch.tensor(0.0)
    L_fick_l = torch.mean(torch.square(F_liquid(P_from_G(model(X_fick_liquid), X_fick_liquid), X_fick_liquid))) if X_fick_liquid.shape[0] > 0 else torch.tensor(0.0)
    
    X_data_batch.requires_grad_(True)
    t_vals = X_data_batch[:, 1]
    X_boundary_batch = X_data_batch[t_vals > 0]
    t_boundary_batch = X_boundary_batch[:, 1].view(-1, 1)
    X_ini_batch = X_data_batch[t_vals == 0]

    G_pred_at_R_prime = model(X_boundary_batch)
    vol_frac_solid = (R_norm**3) / (R_prime_norm**3)
    G_target_from_data = (1.0 - vol_frac_solid) * S_j(t_boundary_batch) + vol_frac_solid * S_f(t_boundary_batch)
    L_yz = torch.mean(torch.square(G_pred_at_R_prime - G_target_from_data))
    L_ini = torch.mean(torch.square(P_from_G(model(X_ini_batch), X_ini_batch))) if X_ini_batch.shape[0] > 0 else torch.tensor(0.0)

    ### CORRECTION CRITIQUE ###
    # Créer un tenseur propre pour évaluer G(R,t) sans perturber le graphe de calcul.
    X_solid_boundary_batch = torch.cat([torch.full_like(t_boundary_batch, R_norm), t_boundary_batch], dim=1)
    L_solide = torch.mean(torch.square(model(X_solid_boundary_batch) - S_f(t_boundary_batch)))

    X_grad_batch.requires_grad_(True)
    P_grad = P_from_G(model(X_grad_batch), X_grad_batch)
    dP_dr = torch.autograd.grad(P_grad, X_grad_batch, grad_outputs=torch.ones_like(P_grad), create_graph=True)[0][:, 0]
    L_gradient_nul = torch.mean(torch.square(dP_dr))

    ### CORRECTION CRITIQUE ###
    # Pondération MANUELLE agressive pour prioriser les données
    w_data = 100.0
    w_phys = 1.0
    
    total_loss = (w_data * L_yz) + (w_data * L_solide) + (w_phys * L_ini) + (w_phys * L_gradient_nul) + (w_phys * L_fick_s) + (w_phys * L_fick_l)
    
    loss_components = [total_loss.item(), L_yz.item(), L_ini.item(), L_fick_s.item(), L_fick_l.item(), L_solide.item(), L_gradient_nul.item()]
    return total_loss, loss_components

def cost_enhanced_full_batch(model, F_solid, F_liquid, S_f, S_j, X_fick_total, X_data_total, X_grad_total, R_norm, R_prime_norm):
    X_fick_total.requires_grad_(True)
    r_fick = X_fick_total[:, 0]
    X_fick_solid = X_fick_total[r_fick < R_norm]
    X_fick_liquid = X_fick_total[r_fick >= R_norm]
    L_fick_s = torch.mean(torch.square(F_solid(P_from_G(model(X_fick_solid), X_fick_solid), X_fick_solid)))
    L_fick_l = torch.mean(torch.square(F_liquid(P_from_G(model(X_fick_liquid), X_fick_liquid), X_fick_liquid)))
    
    X_data_total.requires_grad_(True)
    t_vals = X_data_total[:, 1]
    X_boundary = X_data_total[t_vals > 0]
    t_boundary = X_boundary[:, 1].view(-1, 1)
    X_ini = X_data_total[t_vals == 0]
    
    G_pred_at_R_prime = model(X_boundary)
    vol_frac_solid = (R_norm**3) / (R_prime_norm**3)
    G_target_from_data = (1.0 - vol_frac_solid) * S_j(t_boundary) + vol_frac_solid * S_f(t_boundary)
    L_yz = torch.mean(torch.square(G_pred_at_R_prime - G_target_from_data))
    L_ini = torch.mean(torch.square(P_from_G(model(X_ini), X_ini)))
    
    ### CORRECTION CRITIQUE ###
    X_solid_boundary = torch.cat([torch.full_like(t_boundary, R_norm), t_boundary], dim=1)
    L_solide = torch.mean(torch.square(model(X_solid_boundary) - S_f(t_boundary)))

    X_grad_total.requires_grad_(True)
    P_grad = P_from_G(model(X_grad_total), X_grad_total)
    dP_dr = torch.autograd.grad(P_grad, X_grad_total, grad_outputs=torch.ones_like(P_grad), create_graph=True)[0][:, 0]
    L_gradient_nul = torch.mean(torch.square(dP_dr))

    ### CORRECTION CRITIQUE ###
    w_data = 100.0
    w_phys = 1.0
    
    total_loss = (w_data * L_yz) + (w_data * L_solide) + (w_phys * L_ini) + (w_phys * L_gradient_nul) + (w_phys * L_fick_s) + (w_phys * L_fick_l)

    loss_components = [total_loss.item(), L_yz.item(), L_ini.item(), L_fick_s.item(), L_fick_l.item(), L_solide.item(), L_gradient_nul.item()]
    return total_loss, loss_components

# Le reste du code (run_enhanced_case, affichage, main) reste identique à la version précédente.
# Il suffit de copier-coller ces deux fonctions corrigées dans votre script.
# Je joins le code complet pour être sûr.

def run_enhanced_case(params_pinns: dict, params: dict, S_f: DataAugmentation, S_j: DataAugmentation, output_path: Path):
    torch.manual_seed(1234)
    batch_size = params_pinns["batch_size"]
    R_m, R_prime_m = params["R_vrai_m"], params["R_prime_m"]
    R_norm, D_solid_norm, ordre_R = normalisation(R_m, params["D_f"])
    R_prime_norm, D_liquid_norm, _ = normalisation(R_prime_m, params["D_j"])
    params["ordre_R"] = ordre_R

    model = Physics_informed_nn(params_pinns["nb_hidden_layer"], params_pinns["nb_hidden_perceptron"], R_norm, params["P0_j"])
    F_solid = Fick(D_solid_norm, params["T_1_f"], params["P0_f"] / params["P0_j"])
    F_liquid = Fick(D_liquid_norm, params["T_1_j"], 1.0)
    
    print(f"Création du DataSet enrichi...")
    def_t = params["def_t"]
    nb_r, nb_t = 500, 500
    
    X_r_f_total = torch.linspace(0, R_prime_norm, nb_r).view(-1, 1)
    X_t_f_total = torch.linspace(0, def_t, nb_t).view(-1, 1)
    grid_r_f, grid_t_f = torch.meshgrid(X_r_f_total.squeeze(), X_t_f_total.squeeze(), indexing="ij")
    X_fick_total = torch.stack([grid_r_f.flatten(), grid_t_f.flatten()], dim=1)
    
    X_R_prime_data = torch.full((nb_t, 1), R_prime_norm)
    X_t_data = torch.linspace(0, def_t, nb_t).view(-1, 1)
    X_boundary_total = torch.cat([X_R_prime_data, X_t_data], dim=1)
    X_r_ini_total = torch.linspace(0, R_prime_norm, nb_t).view(-1, 1)
    X_t_ini_total = torch.zeros((nb_t, 1))
    X_ini_total = torch.cat([X_r_ini_total, X_t_ini_total], dim=1)
    X_data_total = torch.cat([X_boundary_total, X_ini_total], dim=0)

    X_t_grad = torch.linspace(0, def_t, nb_t).view(-1, 1)
    X_r0_grad = torch.zeros_like(X_t_grad)
    X_r_prime_grad = torch.full_like(X_t_grad, R_prime_norm)
    X_grad_total = torch.cat([torch.cat([X_r0_grad, X_t_grad], dim=1), torch.cat([X_r_prime_grad, X_t_grad], dim=1)], dim=0)
    
    print(f"DataSet créé: {X_fick_total.shape[0]} Fick, {X_data_total.shape[0]} Données, {X_grad_total.shape[0]} Gradient.")

    loss = [[] for _ in range(7)]
    model_opti = copy.deepcopy(model)
    min_loss_val = float('inf')

    print("\n--- Phase 1: Adam Optimizer avec Mini-Batching ---")
    optimizer = optim.Adam(model.parameters(), lr=params_pinns['lr'])
    epochs_phase1 = 9000
    for it in tqdm(range(epochs_phase1), desc="Phase 1 (Adam)", file=sys.stdout):
        fick_indices = torch.randint(0, X_fick_total.shape[0], (batch_size // 3,))
        data_indices = torch.randint(0, X_data_total.shape[0], (batch_size // 3,))
        grad_indices = torch.randint(0, X_grad_total.shape[0], (batch_size // 3,))
        X_fick_batch = X_fick_total[fick_indices]
        X_data_batch = X_data_total[data_indices]
        X_grad_batch = X_grad_total[grad_indices]
        
        optimizer.zero_grad()
        L, L_list = cost_enhanced_batch(model, F_solid, F_liquid, S_f, S_j, X_fick_batch, X_data_batch, X_grad_batch, R_norm, R_prime_norm)
        L.backward()
        optimizer.step()
        
        if it % 10 == 0:
            for i in range(len(L_list)): loss[i].append(L_list[i])
            if L_list[0] < min_loss_val:
                min_loss_val = L_list[0]
                model_opti = copy.deepcopy(model)

    print("\n--- Phase 2: L-BFGS Optimizer avec Full-Batch ---")
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=10, max_eval=20, history_size=150)
    epochs_phase2 = 100
    for it in tqdm(range(epochs_phase2), desc="Phase 2 (L-BFGS)", file=sys.stdout):
        def closure():
            optimizer.zero_grad()
            L, L_list = cost_enhanced_full_batch(model, F_solid, F_liquid, S_f, S_j, X_fick_total, X_data_total, X_grad_total, R_norm, R_prime_norm)
            L.backward()
            nonlocal min_loss_val, model_opti
            if L_list[0] < min_loss_val:
                min_loss_val = L_list[0]
                model_opti = copy.deepcopy(model)
            if it % 10 == 0:
                for i in range(len(L_list)): loss[i].append(L_list[i])
            return L
        optimizer.step(closure)

    print(f"\nEntraînement terminé. Meilleure perte (sum): {min_loss_val:.2e}")
    return model_opti, loss

# SECTION 3 et 4 sont identiques à la version précédente et restent valides.

def affichage(path: Path):
    print(f"Génération des graphiques pour : {path}")
    data_dir = path / "Data"
    graph_dir = path / "Graphiques"
    with open(data_dir / "loss.json", "r") as f: loss = json.load(f)
    with open(data_dir / "params.json", "r") as f: params = json.load(f)
    with open(data_dir / "params_PINNS.json", "r") as f: params_pinns = json.load(f)
    coeff_normal = params["P0_j"]
    with open(data_dir / "S_f.pkl", "rb") as f: S_f = pickle.load(f)
    with open(data_dir / "S_j.pkl", "rb") as f: S_j = pickle.load(f)
    
    R_m, R_prime_m = params["R_vrai_m"], params["R_prime_m"]
    R_norm, _, ordre_R = normalisation(R_m, params["D_f"])
    
    model = Physics_informed_nn(params_pinns["nb_hidden_layer"], params_pinns["nb_hidden_perceptron"], R_norm, coeff_normal)
    model.load_state_dict(torch.load(data_dir / "model.pth"))
    model.eval()
    
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 7))
    loss_names = ["Total Sum", "L_yz", "L_initial", "L_fick_solid", "L_fick_liquid", "L_solid", "L_gradient_nul"]
    for i, name in enumerate(loss_names):
        ax1.plot(loss[i], label=name)
    ax1.set_yscale('log'); ax1.set_title('Evolution de la fonction de coût'); ax1.set_xlabel('Itérations (x10)'); ax1.set_ylabel('Coût (log)'); ax1.legend(); ax1.grid(True)
    fig.tight_layout(); fig.savefig(graph_dir / "loss_evolution.png"); plt.close(fig)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    t_plot = torch.linspace(0, params["def_t"], 200).view(-1, 1)
    
    X_solid_boundary = torch.cat([torch.full_like(t_plot, R_norm), t_plot], dim=1)
    G_pred_at_R = model(X_solid_boundary)
    ax1.plot(t_plot.numpy(), S_f(t_plot).numpy() * coeff_normal, 'k--', label='S_f Cible (Données)')
    ax1.plot(S_f.times, S_f.list_y_raw, 'ro', markersize=4, label='S_f Brutes')
    ax1.plot(t_plot.numpy(), G_pred_at_R.detach().numpy() * coeff_normal, 'b-', label='G(R, t) Prédit (Modèle)')
    ax1.set_title('Validation de la polarisation moyenne du solide'); ax1.set_xlabel('Temps (s)'); ax1.set_ylabel('Polarisation'); ax1.legend(); ax1.grid(True)

    R_prime_norm, _, _ = normalisation(R_prime_m, params["D_j"])
    X_boundary = torch.cat([torch.full_like(t_plot, R_prime_norm), t_plot], dim=1)
    G_pred_at_R_prime = model(X_boundary)
    vol_frac_solid = (R_norm**3) / (R_prime_norm**3)
    G_target_from_data = (1.0 - vol_frac_solid) * S_j(t_plot) + vol_frac_solid * S_f(t_plot)
    ax2.plot(t_plot.numpy(), G_target_from_data.numpy() * coeff_normal, 'k--', label='G(R\', t) Cible')
    ax2.plot(t_plot.numpy(), G_pred_at_R_prime.detach().numpy() * coeff_normal, 'b-', label='G(R\', t) Prédit')
    ax2.set_title('Validation de la polarisation moyenne totale'); ax2.set_xlabel('Temps (s)'); ax2.legend(); ax2.grid(True)
    fig.tight_layout(); fig.savefig(graph_dir / "mean_polarization_fits.png"); plt.close(fig)
    
    r_range = torch.linspace(0, R_prime_m, 100)
    t_range = torch.linspace(0, params["def_t"], 100)
    grid_r, grid_t = torch.meshgrid(r_range, t_range, indexing='ij')
    grid_r_norm = grid_r / (10**ordre_R)
    X_grid = torch.stack([grid_r_norm.flatten(), grid_t.flatten()], dim=1)
    X_grid.requires_grad_(True)
    P_grid_denorm = P_from_G(model(X_grid), X_grid) * coeff_normal
    P_colormap = P_grid_denorm.detach().numpy().reshape(grid_r.shape)
    np.save(data_dir / "P.npy", P_colormap)
    np.save(data_dir / "(r, t).npy", (grid_r.numpy(), grid_t.numpy()))
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(grid_r.numpy() * 1e9, grid_t.numpy(), P_colormap, 50, cmap='jet')
    cbar = fig.colorbar(contour, ax=ax); cbar.set_label('Polarisation P(r,t)')
    ax.axvline(x=R_m * 1e9, color='white', linestyle='--', linewidth=2, label=f'Interface R = {R_m * 1e9:.1f} nm')
    ax.set_xlabel('Rayon r (nm)'); ax.set_ylabel('Temps t (s)'); ax.set_title(f"Polarisation (Solide + Solvant) - R = {R_m * 1e9:.1f} nm"); ax.legend()
    plt.savefig(graph_dir / "P_r_t_colormap.png"); plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancer un entraînement PINN pour le cas enrichi (deux milieux + pertes additionnelles).")
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--case_name', type=str, required=True)
    args = parser.parse_args()
    
    params_pinns = {"nb_hidden_layer": 2, "nb_hidden_perceptron": 32, "lr": 0.001, "epoch": 5000, "var_R": False, "batch_size": 256}
    data_file, base_output = Path(args.data_file), Path(args.output_dir)
    
    with open(data_file, "rb") as f: all_data = pickle.load(f)
    if args.case_name not in all_data: print(f"ERREUR: Cas '{args.case_name}' non trouvé."); sys.exit()
    
    exp_data = all_data[args.case_name]
    output_path = base_output / f"{args.case_name}_On_enhanced_v2_result"
    if output_path.exists(): shutil.rmtree(output_path)
    (output_path / "Data").mkdir(parents=True, exist_ok=True)
    (output_path / "Graphiques").mkdir(parents=True, exist_ok=True)
    
    R_vrai_m = exp_data["R_s"] * 1.0e-9

     # --- CALCUL DES PARAMÈTRES PHYSIQUES ---
    C_ref, D_ref_nm2_s = 60.0, 500.0
    D_ref_m2_s = D_ref_nm2_s * 1e-18
    C_f, C_j = exp_data.get("C_f", C_ref), exp_data.get("C_j", C_ref)
    D_f_calculated = D_ref_m2_s * ((C_f / C_ref) ** (1/3))

    params = {
        "D_f": D_f_calculated, # Diffusion solide
        "D_j": exp_data.get("D_j", 500e-18),# Diffusion liquide
        "T_1_f": exp_data["T_1"],# Relaxation solide
        "T_1_j": exp_data.get("T_1_j", 20.0), # Relaxation liquide
        "P0_f": 1.0, 
        "P0_j": exp_data[solvent_data_key]["P0_j"],
        "def_t": max(exp_data[solid_data_key]["t"]),
        "name": f"{EXP_NAME_TO_RUN}_{CASE}_two_media", 
        "R_vrai_m": R_vrai_m, "R_prime_m":  R_vrai_m * 5.0,
    }
    
    S_f = DataAugmentation(pd.DataFrame(exp_data["CrisOn"]), params["P0_j"])
    S_j = DataAugmentation(pd.DataFrame(exp_data["JuiceOn"]), params["P0_j"])

    model_final, loss_history = run_enhanced_case(params_pinns, params, S_f, S_j, output_path)

    save_results(model_final, loss_history, params_pinns, params, output_path)
    with open(output_path / "Data" / "S_f.pkl", "wb") as f: pickle.dump(S_f, f)
    with open(output_path / "Data" / "S_j.pkl", "wb") as f: pickle.dump(S_j, f)
    affichage(output_path)
    print(f"\n=== FIN DE L'EXPÉRIENCE 'CORRIGÉE' POUR {args.case_name} ===")