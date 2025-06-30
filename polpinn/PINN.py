import torch
import torch.optim as optim
from tqdm import tqdm
import copy
import shutil

# Importer les outils nécessaires depuis tool.py
from polpinn.tool import (
    Fick,
    normalisation,
    Physics_informed_nn,
    P,
)
# Importer les fonctions de sauvegarde
from polpinn.save import check, save, affichage

def cost(
    model, 
    F_f,        # Physique du solide
    F_j,        # Physique du solvant (Nouveau)
    S_f,        # Données du solide (objet data_augmentation)
    S_j,        # Données du solvant (objet data_augmentation)
    def_t,      # Durée de la simulation
    R_max_norm  # Rayon max normalisé du domaine de collocation (Nouveau)
):
    """
    Calcule la fonction de coût totale pour le PINN.
    """
    R_norm = model.R  # Le rayon est déjà normalisé dans le modèle
    
    # --- 1. Loss sur les données expérimentales (Conditions aux limites de type Dirichlet) ---

    # L_solide : La sortie du modèle G(R,t) doit correspondre à la polarisation moyenne S_f(t)
    nb_t_data = 200 # Nombre de points temporels pour comparer avec les données
    t_data = torch.linspace(0, def_t, nb_t_data).view(-1, 1)
    r_data = R_norm.repeat(nb_t_data, 1).view(-1, 1)
    X_solide_data = torch.cat([r_data, t_data], dim=1)
    
    # On normalise les données S_f pour les comparer à la sortie du réseau (qui est entre 0 et 1)
    sf_values_norm = torch.tensor(S_f(t_data.detach().numpy()), dtype=torch.float32) / model.coeff_normal
    L_solide = torch.mean(torch.square(model(X_solide_data) - sf_values_norm))

    # L_bord : La polarisation P(R,t) doit correspondre à la polarisation du solvant S_j(t)
    # NOTE: C'est la condition que nous allons remplacer par la condition de flux plus tard.
    # On évalue P juste à l'extérieur de R pour être dans le domaine du solvant.
    r_bord_data = (R_norm + 1e-4).repeat(nb_t_data, 1).view(-1, 1) # Petite distance de R
    X_bord_data = torch.cat([r_bord_data, t_data], dim=1)
    X_bord_data.requires_grad_(True)
    
    sj_values_norm = torch.tensor(S_j(t_data.detach().numpy()), dtype=torch.float32) / model.coeff_normal
    P_at_border = P(model(X_bord_data), X_bord_data)
    L_bord = torch.mean(torch.square(P_at_border - sj_values_norm))
    
    # --- 2. Loss sur la condition initiale ---

    # L_ini : P(r, 0) = 0 pour tout r
    nb_r_ini = 100
    r_ini = torch.linspace(0, R_max_norm, nb_r_ini).view(-1, 1)
    t_ini = torch.zeros_like(r_ini)
    X_ini = torch.cat([r_ini, t_ini], dim=1)
    X_ini.requires_grad_(True)
    
    # La condition initiale est sur P, pas sur G.
    L_ini = torch.mean(torch.square(P(model(X_ini), X_ini)))

    # --- 3. Loss sur la physique (Résidu de l'EDP) ---

    # Points de collocation dans tout le domaine [0, R_max] x [0, def_t]
    nb_r_pde = 50
    nb_t_pde = 50
    r_pde = torch.linspace(0, R_max_norm, nb_r_pde).view(-1, 1)
    t_pde = torch.linspace(0, def_t, nb_t_pde).view(-1, 1)
    grid_r, grid_t = torch.meshgrid(r_pde.squeeze(), t_pde.squeeze(), indexing="ij")
    X_collocation = torch.stack([grid_r.flatten(), grid_t.flatten()], dim=1)
    X_collocation.requires_grad_(True)

    # Séparer les points intérieurs et extérieurs
    r_coords = X_collocation[:, 0]
    interior_mask = r_coords < R_norm
    
    X_f = X_collocation[interior_mask]  # Points dans le solide
    X_j = X_collocation[~interior_mask] # Points dans le solvant

    # Calcul du résidu de l'EDP dans chaque domaine
    L_fick_f = torch.tensor(0.0)
    if X_f.shape[0] > 0:
        P_f = P(model(X_f), X_f)
        L_fick_f = torch.mean(torch.square(F_f(P_f, X_f)))

    L_fick_j = torch.tensor(0.0)
    if X_j.shape[0] > 0:
        P_j = P(model(X_j), X_j)
        L_fick_j = torch.mean(torch.square(F_j(P_j, X_j)))
        
    L_fick = L_fick_f + L_fick_j

    # --- 4. Combinaison des pertes avec pondération dynamique ---
    
    # (Tu avais une pondération de type 'gamma', je la conserve)
    total_loss_unweighted = L_solide + L_bord + L_ini + L_fick
    
    # Pour éviter la division par zéro si la loss est nulle
    if total_loss_unweighted > 0:
        gamma_solide = L_solide / total_loss_unweighted
        gamma_bord = L_bord / total_loss_unweighted
        gamma_ini = L_ini / total_loss_unweighted
        gamma_fick = L_fick / total_loss_unweighted
    else:
        gamma_solide = gamma_bord = gamma_ini = gamma_fick = torch.tensor(0.25)
    
    # Loss finale pondérée
    L_total_weighted = (
        gamma_solide * L_solide +
        gamma_bord * L_bord +
        gamma_ini * L_ini +
        gamma_fick * L_fick
    )
    
    # Retourner la loss pondérée pour la backpropagation et les losses individuelles pour le suivi
    loss_components = [total_loss_unweighted, L_solide, L_bord, L_ini, L_fick, L_fick_f, L_fick_j]
    return L_total_weighted, loss_components


def run(
    params_pinns: dict,
    params: dict,
    output_path,
    S_f,  # NOUVEAU: objet data_augmentation pour le solide
    S_j,  # NOUVEAU: objet data_augmentation pour le solvant
    seed=1234,
    update_progress=None,
    no_interaction=False,
    no_gui=False,
):
    check(params=params, path=output_path)
    
    # Le coefficient de normalisation est la valeur maximale attendue (P0 du solvant)
    coeff_normal = params["P0_j"]
    var_R = params_pinns["var_R"]
    
    # Normalisation des unités pour la stabilité numérique
    rayon_ini_norm, D_f_norm, ordre_R = normalisation(
        params["rayon_initialisation"], params["D_f"]
    )
    # On doit aussi normaliser D_j avec le même facteur d'échelle
    _, D_j_norm, _ = normalisation(
        params["rayon_initialisation"], params["D_j"]
    )
    params["ordre_R"] = ordre_R # Sauvegarde pour dénormaliser le rayon final

    # Définition du domaine de collocation étendu (normalisé)
    # Par exemple, on prend un domaine 2 fois plus grand que le rayon initial
    R_max_norm = 2.0 * rayon_ini_norm

    # Initialisation du modèle
    torch.manual_seed(seed)
    model = Physics_informed_nn(
        nb_layer=params_pinns["nb_hidden_layer"],
        hidden_layer=params_pinns["nb_hidden_perceptron"],
        rayon_ini=rayon_ini_norm,
        coeff_normal=coeff_normal,
        var_R=var_R,
    )

    # Instanciation des objets physiques pour chaque domaine
    F_f = Fick(D_f_norm, params["T_1"], params["P0_f"] / coeff_normal)
    F_j = Fick(D_j_norm, params["T_B"], params["P0_j"] / coeff_normal)
    
    # Configuration de l'optimiseur
    if var_R:
        # Apprentissage en 2 temps (curriculum learning)
        # Au début, on ne fait pas varier R pour stabiliser l'apprentissage de la forme de la fonction
        params_without_R = [p for name, p in model.named_parameters() if name != "R"]
        optimizer = optim.Adam(
            [{"params": params_without_R}, {"params": model.R, "lr": 0.0}],
            lr=params_pinns["lr"],
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=params_pinns["lr"])

    # Initialisation des listes pour le suivi des pertes
    loss_names = ["L_total", "L_solide", "L_bord", "L_ini", "L_fick", "L_fick_f", "L_fick_j"]
    if var_R:
        loss_names.append("R")
    losses = {name: [] for name in loss_names}
    
    model_opti = copy.deepcopy(model)
    min_loss_total = float('inf')

    # Boucle d'entraînement
    for it in tqdm(range(params_pinns["epoch"]), desc="Entraînement du PINN"):
        # Activation de l'optimisation de R après une phase de stabilisation
        if var_R and it == 3000:
            print("\nActivation de l'optimisation du rayon R...")
            params_without_R = [p for name, p in model.named_parameters() if name != "R"]
            optimizer = optim.Adam(
                [
                    {"params": params_without_R},
                    {"params": model.R, "lr": params_pinns["lr_R"]},
                ],
                lr=params_pinns["lr"],
            )

        optimizer.zero_grad()

        L_weighted, L_components = cost(model, F_f, F_j, S_f, S_j, params["def_t"], R_max_norm)
        
        # Sauvegarde des valeurs de loss pour le suivi
        for i, name in enumerate(loss_names):
            if name != "R":
                losses[name].append(L_components[i].item())
        if var_R:
            losses["R"].append(model.R.item())

        # Sauvegarde du meilleur modèle
        if L_components[0].item() < min_loss_total:
            min_loss_total = L_components[0].item()
            model_opti = copy.deepcopy(model)

        L_weighted.backward()
        optimizer.step()
        
        # Mise à jour de l'affichage en temps réel si une fonction est fournie
        if update_progress is not None and it % 10 == 0:
            R_denorm = model_opti.R.item() * 10**(ordre_R + 1)
            update_progress(value=it + 1, best_R=R_denorm, best_loss=min_loss_total)

    # Conversion du dictionnaire de losses en liste de listes pour la sauvegarde JSON
    loss_list_for_json = [losses[name] for name in loss_names]
    
    # Dénormalisation du rayon final pour la sauvegarde
    params["R_final_m"] = model_opti.R.item() * 10**(ordre_R + 1)
    
    print(f"\nEntraînement terminé. Meilleure loss totale : {min_loss_total:.4e}")
    print(f"Rayon final estimé : {params['R_final_m']:.4e} m")
    
    save(model_opti, loss_list_for_json, params_pinns, params, output_path)
    affichage(output_path)