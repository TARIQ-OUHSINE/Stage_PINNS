# PINN.py (version complète et corrigée pour l'Axe 1)

import torch
import torch.optim as optim
import copy
import shutil
from tqdm import tqdm

# On importe les outils nécessaires. 
# Assure-toi que ton tool.py contient bien PINN_Axe1 et Fick_Discontinuous.
from polpinn.tool import (
    normalisation,
    data_augmentation,
    PINN_Axe1,
    Fick_Discontinuous,
    hypo,
)
from polpinn.savee import check, save, affichage

# --- NOUVELLE FONCTION DE COUT (AXE 1) ---
def cost(model, F_phys, S_f, S_j, def_t, R_max_norm, coeff_normal):
    """
    Calcule la loss totale pour le modèle de l'Axe 1.
    Le modèle prédit P(r,t) sur un domaine étendu [0, R_max].
    """
    R_norm = model.R  # Le rayon normalisé, qui est entraînable

    # --- 1. Loss Physique (Résidu de l'EDP) ---
    # Points de collocation dans tout le domaine [0, R_max]
    nb_r_phys = 100
    nb_t_phys = 100
    # On échantillonne r de manière non-uniforme pour avoir plus de points près de l'interface R
    # C'est une heuristique qui aide souvent. r^2 assure plus de points vers R_max.
    r_phys = torch.rand(nb_r_phys, 1)**2 * R_max_norm
    r_phys[0] = 1e-7 # pour éviter la division par zéro dans le laplacien
    t_phys = torch.linspace(0, def_t, nb_t_phys).view(-1, 1)
    
    grid_r, grid_t = torch.meshgrid(r_phys.squeeze(), t_phys.squeeze(), indexing='ij')
    X_phys = torch.stack([grid_r.flatten(), grid_t.flatten()], dim=1)
    X_phys.requires_grad_(True)
    
    P_pred_norm = model(X_phys)
    residu = F_phys(P_pred_norm, X_phys, R_norm)
    L_fick = torch.mean(torch.square(residu))

    # --- 2. Loss sur les Données du Solide (S_f) ---
    # S_f est la polarisation moyenne du solide, calculée par intégrale de P(r,t)
    t_data = torch.linspace(0, def_t, 50).view(-1, 1)
    
    # Intégration de P sur le volume de la sphère de rayon R via Monte Carlo
    nb_int_pts = 256 # Plus de points pour une meilleure précision
    
    # Points aléatoires dans la sphère de rayon R
    r_int_solide = torch.rand(nb_int_pts, 1) * R_norm
    
    # On calcule la polarisation moyenne prédite pour chaque instant t_data
    P_moy_pred_list = []
    for t_i in t_data:
        t_int = t_i.repeat(nb_int_pts, 1)
        X_int = torch.cat([r_int_solide, t_int], dim=1)
        
        # Le modèle prédit P normalisé, on le remet à l'échelle
        P_vals = model(X_int) * coeff_normal
        
        # Moyenne = (3/R³) * ∫ P(r) r² dr
        # Approximation Monte Carlo: 3 * mean(P_i * r_i²) / R³
        # On ajoute un epsilon pour éviter la division par zéro si R est très petit
        P_moy_pred = 3.0 * torch.mean(P_vals * r_int_solide**2) / (R_norm**3 + 1e-9)
        P_moy_pred_list.append(P_moy_pred)
        
    P_moy_pred_tensor = torch.stack(P_moy_pred_list)
    S_f_target = S_f(t_data)
    L_solide = torch.mean(torch.square(P_moy_pred_tensor.squeeze() - S_f_target.squeeze()))

    # --- 3. Loss sur la Condition Initiale ---
    r_ini = torch.linspace(0, R_max_norm, nb_r_phys).view(-1, 1)
    t_ini = torch.zeros_like(r_ini)
    X_ini = torch.cat([r_ini, t_ini], dim=1)
    P_ini_pred_norm = model(X_ini)
    L_ini = torch.mean(torch.square(P_ini_pred_norm))  # P(r,0) doit être 0

    # --- Loss Totale ---
    # On utilise des poids fixes pour commencer. C'est plus stable et facile à déboguer.
    # Donner plus de poids aux données est crucial.
    lambda_data = 100.0
    loss_totale = L_fick + lambda_data * L_solide + L_ini
    
    return loss_totale, [loss_totale.item(), L_fick.item(), L_solide.item(), L_ini.item()]


# --- FONCTION PRINCIPALE `run` (ADAPTÉE) ---
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
    # CORRECTION : Création des dossiers de sortie au début.
    check(path=output_path)

    torch.manual_seed(seed)
    
    # --- Paramètres et Normalisation ---
    # NOUVEAU : Définir un domaine max pour r. Il doit être plus grand que le R attendu.
    # Par exemple 1.5 à 2 fois le rayon max que tu penses tester.
# --- Paramètres et Normalisation (SECTION CORRIGÉE) ---
    R_max = 800e-9 
    params['R_max'] = R_max
    
    coeff_normal = params["P0_j"]
    var_R = params_pinns["var_R"]

    # On normalise R_init et on récupère son ordre de grandeur.
    # Cet ordre de grandeur sera LA référence pour tout le reste.
    rayon_ini_norm, ordre_R = normalisation(params["rayon_initialisation"])
    params["ordre_R"] = ordre_R # On sauvegarde l'ordre de grandeur de référence

    # On normalise les autres valeurs en utilisant cet ordre de grandeur de référence
    R_max_norm = R_max / (10**ordre_R)
    
    # Le coefficient de diffusion D est en m²/s. Il doit être normalisé par (longueur_norm)²
    # donc par (10**ordre_R)².
    facteur_norm_D = (10**ordre_R)**2
    D_f_norm = params["D_f"] / facteur_norm_D
    D_j_norm = params["D_j"] / facteur_norm_D

    # --- Le reste de la fonction run ---
    torch.manual_seed(seed)
    
    model = PINN_Axe1(
        nb_layer=params_pinns["nb_hidden_layer"],
        hidden_layer=params_pinns["nb_hidden_perceptron"],
        rayon_ini=rayon_ini_norm,
        var_R=var_R,
    )

    params_solide_norm = {'D': D_f_norm, 'T1': params['T_1'], 'P0': params['P0_f'] / coeff_normal}
    params_solvant_norm = {'D': D_j_norm, 'T1': params['T_B'], 'P0': params['P0_j'] / coeff_normal}
    F_phys = Fick_Discontinuous(params_solide_norm, params_solvant_norm)
    
    # --- Préparation des Données ---
    # La logique de data_augmentation ne change pas
    S_f = data_augmentation(data_path, output_path, "S_f", params["name"])
    S_j = data_augmentation(data_path, output_path, "S_j", params["name"])
    S_j_mono = data_augmentation(data_path, output_path, "S_j", params["name"], mono=True)

    # La fonction hypo est utile pour visualiser les données d'entrée, on la garde.
    if hypo(S_f, S_j, S_j_mono, output_path / "Graphiques" / "input_data_fit.png", no_interaction, no_gui):
        print("Exécution annulée par l'utilisateur après visualisation des données.")
        shutil.rmtree(output_path)
        return

    # --- Configuration de l'Optimiseur ---
    # La stratégie de curriculum learning (activer lr_R plus tard) est excellente, on la garde.
    if var_R:
        params_without_R = [param for name, param in model.named_parameters() if name != "R"]
        optimizer = optim.Adam(
            [{"params": params_without_R}, {"params": model.R, "lr": 0.0}],
            lr=params_pinns["lr"],
        )
    else:
        # Si R n'est pas variable, Adam est un bon choix pour tout.
        optimizer = optim.Adam(model.parameters(), lr=params_pinns["lr"])

    loss_history = []
    model_opti = copy.deepcopy(model) # Initialiser le meilleur modèle
    min_loss_val = float('inf')

    # --- Boucle d'Entraînement ---
    for it in tqdm(range(params_pinns["epoch"]), desc="Training process"):
        # Activation de l'apprentissage de R après une phase de pré-entraînement
        if it == 3000 and var_R:
            print("\n--- Activation de l'optimisation de R ---")
            params_without_R = [param for name, param in model.named_parameters() if name != "R"]
            optimizer = optim.Adam(
                [
                    {"params": params_without_R, "lr": params_pinns["lr"]},
                    {"params": model.R, "lr": params_pinns["lr_R"]},
                ],
            )
        
        optimizer.zero_grad()

        # Calcul de la loss
        L, L_components = cost(model, F_phys, S_f, S_j, params["def_t"], R_max_norm, coeff_normal)
        
        # Stockage de l'historique
        if var_R:
            loss_history.append(L_components + [model.R.item()])
        else:
            loss_history.append(L_components)

        # Sauvegarde du meilleur modèle
        if L_components[0] < min_loss_val:
            min_loss_val = L_components[0]
            model_opti = copy.deepcopy(model)

        L.backward()
        optimizer.step()

        # S'assurer que R reste dans des bornes physiques [0, R_max_norm]
        if var_R:
            with torch.no_grad():
                model.R.clamp_(0, R_max_norm)

        # Mise à jour pour l'affichage (si GUI)
        if update_progress is not None and it % 10 == 0:
            current_R_denorm = model_opti.R.item() * 10**(ordre_R + 1)
            update_progress(value=it + 1, best_R=current_R_denorm, best_loss=min_loss_val)

    # --- Fin de l'Entraînement et Sauvegarde ---
    final_R_denorm = model_opti.R.item() * 10**(ordre_R + 1)
    params["R_final_estimé"] = final_R_denorm
    print(f"\nEntraînement terminé. Rayon final estimé : {final_R_denorm * 1e9:.2f} nm")

    # Transposer l'historique de loss pour la sauvegarde
    loss_to_save = list(map(list, zip(*loss_history)))
    
    save(copy.deepcopy(model_opti), loss_to_save, params_pinns, params, output_path)
    # Il faudra adapter `affichage` pour ce nouveau modèle.
    # affichage(output_path)
    print(f"Résultats sauvegardés dans : {output_path}")