# tool.py (version complète et corrigée pour l'Axe 1)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import pickle
import json
import os
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Any

# Pas besoin de tkinter ici, on le garde dans les fichiers frontend/save
# import tkinter as tk
# from tkinter import messagebox
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- NOUVELLE CLASSE POUR LE RESEAU DE NEURONES (AXE 1) ---
class PINN_Axe1(nn.Module):
    """
    Réseau de neurones pour l'Axe 1.
    Prédit directement la polarisation ponctuelle P(r,t) normalisée.
    Le rayon R de la sphère est un paramètre entraînable du modèle.
    """
    def __init__(
        self,
        nb_layer: int,
        hidden_layer: int,
        rayon_ini: float,
        var_R: bool,
    ):
        super(PINN_Axe1, self).__init__()
        
        # Construction dynamique du réseau
        layers = [nn.Linear(2, hidden_layer), nn.Tanh()]
        for _ in range(nb_layer):
            layers.extend([nn.Linear(hidden_layer, hidden_layer), nn.Tanh()])
        layers.append(nn.Linear(hidden_layer, 1))
        
        self.net = nn.Sequential(*layers)
        
        if var_R:
            # R est un paramètre que l'optimiseur peut modifier
            self.R = nn.Parameter(
                torch.tensor(float(rayon_ini), dtype=torch.float32, requires_grad=True)
            )
        else:
            # R est un tensor fixe (un buffer)
            self.register_buffer('R', torch.tensor(float(rayon_ini), dtype=torch.float32))

    def forward(self, x):
        """
        La sortie du réseau est directement la polarisation P, mais bornée par une sigmoïde.
        Cela garantit que P_normalisée reste entre 0 et 1, ce qui stabilise l'apprentissage.
        """
        return torch.sigmoid(self.net(x))


# --- NOUVELLE CLASSE POUR L'EQUATION DE FICK (AXE 1) ---
class Fick_Discontinuous:
    """
    Calcule le résidu de l'équation de Fick pour un système avec des coefficients
    discontinus à l'interface r=R.
    """
    def __init__(self, params_solide: dict, params_solvant: dict) -> None:
        self.params_solide = params_solide
        self.params_solvant = params_solvant

    def __call__(self, P_norm: torch.Tensor, X: torch.Tensor, R: torch.Tensor, *args: Any, **kwds: Any) -> Any:
        X.requires_grad_(True)
        X_r = X[:, 0].view(-1, 1)
        
        # Création d'un masque "doux" pour séparer le solide du solvant.
        # sigmoid(-k * (r - R)) tend vers 1 si r < R, et 0 si r > R.
        # k=1000 rend la transition très abrupte, simulant une discontinuité.
        mask = torch.sigmoid(-1000.0 * (X_r - R))

        # Interpolation des paramètres physiques en utilisant le masque
        D  = mask * self.params_solide['D']  + (1 - mask) * self.params_solvant['D']
        T1 = mask * self.params_solide['T1'] + (1 - mask) * self.params_solvant['T1']
        P0_norm = mask * self.params_solide['P0'] + (1 - mask) * self.params_solvant['P0']
        
        # Calcul des dérivées de P_norm par rapport aux entrées (r, t)
        grad_P = torch.autograd.grad(
            P_norm, X, grad_outputs=torch.ones_like(P_norm), create_graph=True
        )[0]
        
        dP_dr = grad_P[:, 0].view(-1, 1)
        dP_dt = grad_P[:, 1].view(-1, 1)
        
        # Calcul de la dérivée seconde pour le Laplacien
        # On dérive dP_dr uniquement par rapport à la coordonnée r (première colonne de X)
        grad_dP_dr = torch.autograd.grad(
            dP_dr, X, grad_outputs=torch.ones_like(dP_dr), create_graph=True
        )[0]
        dP_drr = grad_dP_dr[:, 0].view(-1, 1)

        # Laplacien en coordonnées sphériques pour une fonction ne dépendant que de r
        laplacien_P = dP_drr + (2.0 / (X_r + 1e-9)) * dP_dr # Epsilon pour éviter la division par zéro
        
        # Résidu de l'équation de Fick : ∂P/∂t - D * Laplacien(P) + (P - P0)/T1 = 0
        residu = dP_dt - D * laplacien_P + (P_norm - P0_norm) / T1

        return residu


# --- FONCTIONS UTILITAIRES (INCHANGÉES OU LÉGÈREMENT MODIFIÉES) ---

# tool.py


def normalisation(val_physique: float) -> tuple:
    """
    Normalise une valeur physique et retourne la valeur normalisée et l'ordre de grandeur.
    Ex: R=250e-9 -> ordre=-7, R_norm=2.5
    Retourne (valeur_normalisée, ordre_de_grandeur)
    """
    if val_physique == 0:
        return 0.0, 0

    ordre = math.floor(math.log10(abs(val_physique)))
    val_norm = val_physique / (10**ordre)
    
    return val_norm, ordre

class data_augmentation:
    """
    Classe pour fitter les données expérimentales avec une exponentielle étirée
    et fournir une fonction continue.
    (Code original, semble correct et robuste)
    """
    def __init__(self, data_path, output_path, S, name, mono=False) -> None:
        self.mono = mono
        self.data_path = data_path
        self.output_path = output_path
        self.S = S
        self.name = name
        catExcel = pd.read_excel(data_path / (self.S + ".xlsx"))

        self.times = np.array(catExcel["t"])
        self.list_y = np.array(catExcel["y"])

        self.tau, self.beta, self.C = 1.0, 1.0, max(self.list_y) if len(self.list_y) > 0 else 1.0
        best_params, self.min_loss = self.run()

        if self.mono:
            self.tau, self.C = best_params
            self.beta = 1.0 # beta est 1 pour une exponentielle simple
            path = self.output_path / "Data" / (self.S + "_mono.pkl")
        else:
            self.tau, self.beta, self.C = best_params
            path = self.output_path / "Data" / (self.S + ".pkl")

        with open(path, "wb") as file:
            pickle.dump(self, file)
            
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Permet d'appeler l'objet comme une fonction. Prend un tensor PyTorch."""
        t_np = t.detach().cpu().numpy()
        res_np = self.C * (1 - np.exp(-((t_np / self.tau) ** self.beta)))
        return torch.from_numpy(res_np).float().to(t.device)

    def cost(self, params):
        if self.mono:
            self.tau, self.C = params
            beta_loc = 1.0
        else:
            self.tau, self.beta, self.C = params
            beta_loc = self.beta
        y_pred = self.C * (1 - np.exp(-((self.times / self.tau) ** beta_loc)))
        return np.mean((y_pred - self.list_y) ** 2)

    def callback(self, params):
        self.cost_history.append(float(self.cost(params)))
        self.params_history.append(params.copy())

    def run(self):
        if self.mono:
            initial_params = [self.tau, self.C]
            bounds = [(1e-3, None), (1e-3, None)]
        else:
            initial_params = [self.tau, self.beta, self.C]
            bounds = [(1e-3, None), (0.1, 5.0), (1e-3, None)]

        self.cost_history = []
        self.params_history = []

        result = minimize(
            self.cost,
            initial_params,
            method="L-BFGS-B",
            bounds=bounds,
            callback=self.callback,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if not self.cost_history:
             return result.x, result.fun

        best_params_idx = np.argmin(self.cost_history)
        best_params = self.params_history[best_params_idx]
        min_loss = self.cost_history[best_params_idx]
        return best_params, min_loss

def hypo(S_f, S_j, S_j_mono, path, no_interaction, no_gui=False):
    """Fonction pour visualiser les données d'entrée et leur fit."""
    # (Code original, semble correct)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    min_times = min(S_f.times[-1], S_j.times[-1])
    X_t = torch.linspace(0, min_times, 1000).view(-1, 1)

    # Plot S_f
    ax1.plot(X_t.numpy(), S_f(X_t).numpy(), color="red", linestyle="-", label="Fit S_f (stretch exp)")
    ax1.scatter(S_f.times, S_f.list_y, label="Data points y_i", color="black")
    ax1.set_xlabel("t (s)", fontsize=14)
    ax1.set_ylabel("S_f(t)", fontsize=14)
    ax1.set_title("Polarisation Moyenne du Solide (Données d'entrée)", fontsize=16)
    ax1.grid(True)
    ax1.legend()

    # Plot S_j
    ax2.plot(X_t.numpy(), S_j(X_t).numpy(), color="red", linestyle="-", label="Fit S_j (stretch exp)")
    ax2.plot(X_t.numpy(), S_j_mono(X_t).numpy(), color="green", linestyle="--", label="Fit S_j (mono exp)")
    ax2.scatter(S_j.times, S_j.list_y, label="Data points z_i", color="black")
    ax2.set_xlabel("t (s)", fontsize=14)
    ax2.set_ylabel("S_j(t)", fontsize=14)
    ax2.set_title("Polarisation Moyenne du Solvant (Données d'entrée)", fontsize=16)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(path)
    
    if not no_interaction and not no_gui:
        plt.show(block=True) # Mettre block=True pour attendre la fermeture
    plt.close(fig)

    if no_interaction:
        return False # Continuer automatiquement
    
    # Logique de confirmation
    if no_gui:
        answer = input("Voulez-vous continuer l'entraînement? [Y/n]: ").lower()
        return answer not in ('y', '')
    else:
        # La logique avec Tkinter est complexe à gérer ici,
        # il est plus simple de la laisser dans le frontend.
        # Pour un run en console, la question suffit.
        return False


# --- FONCTIONS DE RECHARGEMENT (ADAPTÉES POUR AXE 1) ---

class PINNS_reload_Axe1(nn.Module):
    """
    Classe miroir de PINN_Axe1 pour recharger un modèle sauvegardé.
    Gère la dé-normalisation pour l'inférence.
    """
    def __init__(
        self,
        nb_layer: int,
        hidden_layer: int,
        var_R: bool,
        ordre_R: int,
        coeff_normal: float,
    ):
        super(PINNS_reload_Axe1, self).__init__()
        self.ordre_R = ordre_R
        self.coeff_normal = coeff_normal
        
        layers = [nn.Linear(2, hidden_layer), nn.Tanh()]
        for _ in range(nb_layer):
            layers.extend([nn.Linear(hidden_layer, hidden_layer), nn.Tanh()])
        layers.append(nn.Linear(hidden_layer, 1))
        self.net = nn.Sequential(*layers)
        
        if var_R:
            self.R = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        # x est supposé être en unités physiques (m, s)
        X_r = x[:, 0].view(-1, 1)
        X_t = x[:, 1].view(-1, 1)

        # Normalisation des entrées
        X_r_norm = X_r / (10**self.ordre_R)
        
        x_norm = torch.cat([X_r_norm, X_t], dim=1)
        
        # Prédiction et dé-normalisation de la sortie
        p_norm = torch.sigmoid(self.net(x_norm))
        return p_norm * self.coeff_normal


def reload_model_Axe1(path: Path):
    """
    Charge un modèle de type Axe 1 sauvegardé.
    """
    path = Path(path)
    if not path.is_dir(): return "Dossier n'existe pas"
    
    model_path = path / "Data" / "model.pth"
    params_path = path / "Data" / "params.json"
    pinns_params_path = path / "Data" / "params_PINNS.json"

    if not all([model_path.exists(), params_path.exists(), pinns_params_path.exists()]):
        return "Fichiers manquants dans le dossier Data."

    with open(params_path, "r") as f:
        params = json.load(f)
    with open(pinns_params_path, "r") as f:
        params_pinns = json.load(f)

    model = PINNS_reload_Axe1(
        nb_layer=params_pinns["nb_hidden_layer"],
        hidden_layer=params_pinns["nb_hidden_perceptron"],
        var_R=params_pinns["var_R"],
        ordre_R=params.get("ordre_R", -7), # Utiliser .get pour la compatibilité
        coeff_normal=params.get("P0_j", 200.0)
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model