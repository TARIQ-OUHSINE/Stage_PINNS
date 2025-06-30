import torch, copy, shutil, os
import torch.optim as optim
from tqdm import tqdm
import copy, shutil
import matplotlib.pyplot as plt

from polpinn.simple_model.data import load_data, data_augmentation
from polpinn.simple_model.model import G_MLP, PointPolarization
# Assurez-vous que le fichier losses.py est bien celui que je vous ai fourni, basé sur le poster
from polpinn.step_model.losses1 import compute_loss 
from polpinn.step_model.utils import step_function
from polpinn.utils import get_data_dir, get_output_dir


def run(name="Moyenne_homo_HyperP", seed=1234, update_progress=None):
    torch.manual_seed(seed)

    # --- MODIFICATION 1 : Ajustement des hyperparamètres ---
    params_pinns = {
        "n_layers": 2,
        "n_neurons_per_layer": 32,
        "lr": 0.001,      # Taux d'apprentissage pour Adam
        "epoch": 1000,   # Augmenté pour donner le temps au modèle de converger
        "var_R": False,
    }

    # --- MODIFICATION 2 : Pondération des pertes pour guider l'optimiseur ---
    # Ces valeurs sont un bon point de départ, n'hésitez pas à les ajuster.
    loss_weights = {
        "pde": 1.0,               # Le poids de base pour la physique
        "conservation_data": 10.0,# Cette contrainte de données est importante
        "init": 100.0,            # La condition initiale doit être fortement respectée
        "symmetry": 100.0,        # La condition de symétrie aussi
    }
    
    # --- Chargement des données (inchangé) ---
    data = load_data(data_path=get_data_dir() / name)
    params = data["params"]
    print(params)

    t_max = params["def_t"]
    rayon_ini = params["rayon_initialisation"]
    system_radius = params["R_bis"]

    # Normalisation du rayon (inchangé)
    normalisation_rayon = system_radius / 0.1
    rayon_ini = rayon_ini / normalisation_rayon
    system_radius = system_radius / normalisation_rayon
    R = torch.tensor(rayon_ini)

    # Fonctions step pour les paramètres physiques (inchangé)
    T = step_function(threshold=rayon_ini, value_before=params["T_1"], value_after=params["T_B"])
    D = step_function(threshold=rayon_ini, value_before=params["D_f"] / normalisation_rayon ** 2, value_after=params["D_j"] / normalisation_rayon ** 2)
    P0 = step_function(threshold=rayon_ini, value_before=params["P0_f_normalized"], value_after=params["P0_j_normalized"])
    
    # Initialisation des modèles (inchangé)
    print("Hyperparamètres du modèle:", params_pinns)
    print("Poids des fonctions de coût:", loss_weights)
    model_G = G_MLP(nb_layer=params_pinns["n_layers"], hidden_layer=params_pinns["n_neurons_per_layer"])
    model_P = PointPolarization(G=model_G)

    # Augmentation des données (inchangé)
    data_aug = {}
    data_aug["average_in"] = data_augmentation(mono=False).fit(**data["average_in"])
    data_aug["average_out"] = data_augmentation(mono=False).fit(**data["average_out"])

    n_samples = 1000
    t_data_original = torch.linspace(0, t_max, n_samples)
    
    # --- MODIFICATION 3 : Normalisation du temps pour les entrées du réseau ---
    t_data_norm = t_data_original / t_max # Le temps en entrée du réseau sera entre 0 et 1

    # Les données pour le calcul de la loss utilisent le temps normalisé
    augmentated_data = {
        k: {"times": t_data_norm, "values": data_aug[k](t_data_original)}
        for k in ("average_in", "average_out")
    }

    # --- MODIFICATION 1 : Changement d'optimiseur pour Adam ---
    optimizer = optim.Adam(model_G.parameters(), lr=params_pinns["lr"])
    losses = {}

    for it in tqdm(range(params_pinns["epoch"]), desc="Training process"):
        optimizer.zero_grad()

        # L'appel à compute_loss utilise maintenant un t_max normalisé (1.0)
        losses_it = compute_loss(
            model_G=model_G,
            model_P=model_P,
            data_in=augmentated_data["average_in"],
            data_out=augmentated_data["average_out"],
            R=R,
            system_radius=system_radius,
            t_max=1.0, # Le temps est normalisé, donc t_max pour le modèle est 1.0
            T=T,
            P0=P0,
            D=D,
            n_samples=n_samples
        )

        # Application des poids à chaque composante de la loss
        losses_it["total"] = sum(loss_weights[k] * losses_it[k] for k in loss_weights)
        
        # Stockage des valeurs scalaires pour le plotting
        for k in losses_it:
            if k not in losses:
                losses[k] = torch.zeros(params_pinns["epoch"])
            losses[k][it] = losses_it[k].item()

        losses_it["total"].backward()
        optimizer.step()

    print("*" * 80)
    print("*" * 35, f'Loss finale : {losses["total"][-1]:.2e}', "*" * 35)
    print("*" * 80)

    out_dir = get_output_dir() / 've_step_corrected' # Nouveau dossier de sortie
    out_dir.mkdir(exist_ok=True)

    # Sauvegarde des plots (les plots de données et de loss ne changent pas)
    fig, ax = plt.subplots(1,2, figsize=(12, 5))
    plt.sca(ax[0])
    data_aug["average_in"].plot()
    ax[0].set_title('Data in')
    plt.sca(ax[1])
    data_aug["average_out"].plot()
    ax[1].set_title('Data out')
    plt.savefig(out_dir / f"{params['name']}_data.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(26, 8))
    for k in losses:
        ax.plot(losses[k], label=k)
    ax.set_yscale("log")
    ax.set_xlabel("itérations", fontsize=21)
    ax.set_ylabel("Coût", fontsize=21)
    ax.set_title("Coût en fonction des itérations", fontsize=23, pad=5)
    ax.tick_params(axis="x", labelsize=19)
    ax.tick_params(axis="y", labelsize=19)
    ax.grid(True)
    ax.legend(fontsize=17)
    plt.savefig(out_dir / f"{params['name']}_losses.pdf")

    # --- MODIFICATION 3 (suite) : Plotting final avec gestion du temps normalisé ---
    # Création de la grille pour le plot final
    X_r_plot = torch.linspace(0, system_radius, 100)
    X_t_norm_plot = torch.linspace(0, 1.0, 100) # La grille pour le modèle utilise le temps normalisé
    
    X_r_mesh, X_t_norm_mesh = torch.meshgrid([X_r_plot, X_t_norm_plot], indexing='ij')
    X_for_model = torch.cat([X_r_mesh.reshape(-1, 1), X_t_norm_mesh.reshape(-1, 1)], dim=1)

    p_tensor = model_P(X_for_model)

    plt.figure(figsize=(12, 8))
    # Pour l'axe Y du graphique, on utilise le temps original (non-normalisé)
    X_t_original_plot = torch.linspace(0, t_max, 100)
    plt.contourf(X_r_plot * normalisation_rayon * 1e9, X_t_original_plot, p_tensor.detach().numpy().reshape(X_r_mesh.shape), 50, cmap="jet")

    cbar = plt.colorbar(label="P(r,t)")
    cbar.ax.yaxis.label.set_size(15)
    cbar.ax.tick_params(labelsize=13)
    plt.xlabel("r (nm)", fontsize=15)
    plt.ylabel("t (s)", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(f"Polarisation ponctuelle d'une sphère de rayon: {params['rayon_initialisation']*10**9:.3f} nm", fontsize=16, pad=16, x=0.6)
    plt.savefig(out_dir / f"{params['name']}_polarisation.pdf")


def run_all():
    for k in get_data_dir().glob('*'):
        print('#' * 80)
        print(k.stem)
        if k.is_dir():
            run(name=k.stem)

def run_selected(names: list):
    for name in names:
        print('#' * 80)
        print(f"Lancement de l'entraînement pour : {name}")
        run(name=name)

if __name__ == "__main__":
    noms_a_lancer = ["Moyenne_homo_HyperP"]
    run_selected(noms_a_lancer)
    plt.show()