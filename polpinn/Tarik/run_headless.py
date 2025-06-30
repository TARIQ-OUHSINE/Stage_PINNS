
from polpinn.PINN import run
from polpinn.save import affichage, cercle
from polpinn.utils import get_output_dir, get_data_dir
from pathlib import Path

params_pinns = {
    "nb_hidden_layer": 2,
    "nb_hidden_perceptron": 32,
    "lr": 0.001,
    "lr_R": 0.0001,
    "epoch": 100,
    "var_R": True,
}

params = {
    "D_f": 5e-16,
    "D_j": 2.5807e-16,
    "T_1": 20,
    "T_B": 3,
    "P0_f": 1,
    "P0_j": 200,
    "R_bis": 225e-9,
    "def_t": 20,
    "rayon_initialisation": 100e-9,
    "name": "SimSansGUI",
}

if __name__ == "__main__":
    data_path = get_data_dir() / "Moyenne_homo_HyperP"
    output_path = get_output_dir() / params["name"]

    print("Entraînement en cours...")
    run(
        params_pinns=params_pinns,
        params=params,
        data_path=data_path,
        output_path=output_path,
        no_interaction=True,
        no_gui=True,
    )

    print("Génération des graphiques...")
    affichage(output_path)

    print("Génération du graphique polaire...")
    cercle(output_path, no_interaction=True)

    print("Terminé !")
