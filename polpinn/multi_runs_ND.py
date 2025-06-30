import json
import shutil
import torch
import pickle
import pandas as pd
from pathlib import Path

# Importer les fonctions nécessaires de ton package polpinn
from polpinn.PINN import run
from polpinn.utils import get_output_dir
from polpinn.tool import data_augmentation
from polpinn.save1 import affichage

# === CONFIGURATION DE L'EXPÉRIENCE ===

# Sélectionne le nom de l'expérience à lancer depuis ton fichier de données.
# Exemples: "11_58_40_25", "110_20_300_50"
EXP_NAME_TO_RUN = "11_58_40_25" 

# Choisis le cas d'étude : "On" pour l'hyperpolarisation, "Off" pour la dépolarisation.
CASE = "On"

# --- Chemins et paramètres globaux ---
root_path = Path(__file__).resolve().parents[2]
data_file = root_path / "data_1" / "donnees.pkl"

# Dossier de base pour les résultats de cette série de tests
base_output = get_output_dir() / "multiple_runs_ND_1"

# Paramètres du réseau de neurones (peuvent être ajustés ici)
params_pinns = {
    "nb_hidden_layer": 2,
    "nb_hidden_perceptron": 32,
    "lr": 0.001,
    "lr_R": 0.0005,
    "epoch": 1000,
    "var_R": False,
}

# === DÉROULEMENT DU SCRIPT ===

# 1. Chargement de toutes les données prétraitées
print(f"Chargement du fichier de données : {data_file}")
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

# Création des dossiers de sortie. La classe data_augmentation en aura besoin.
(output_path / "Data").mkdir(parents=True, exist_ok=True)
(output_path / "Graphiques").mkdir(parents=True, exist_ok=True)


# 3. Création et sauvegarde des objets de données (courbes fittées)
print("Création et sauvegarde des objets S_f et S_j par augmentation de données...")
solid_df = pd.DataFrame(exp_data[solid_data_key])
solvent_df = pd.DataFrame(exp_data[solvent_data_key])

# La classe data_augmentation va automatiquement sauvegarder les fichiers .pkl 
# dans le `output_path` fourni lors de son initialisation.
S_f = data_augmentation(output_path, "S_f", EXP_NAME_TO_RUN, data_df=solid_df)
S_j = data_augmentation(output_path, "S_j", EXP_NAME_TO_RUN, data_df=solvent_df)
S_j_mono = data_augmentation(output_path, "S_j", EXP_NAME_TO_RUN, data_df=solvent_df, mono=True)
print("... Objets S_f, S_j, et S_j_mono créés et sauvegardés.")

# 4. Construction du dictionnaire de paramètres pour le PINN
print("Construction des paramètres physiques pour l'entraînement...")
params = {
    "D_f": 5e-16, "D_j": 5e-16,  # Valeurs à vérifier/ajuster
    "T_1": exp_data["T_1"],
    "T_B": exp_data[solvent_data_key]["TB_j"],
    "P0_f": exp_data[solid_data_key]["P0_j"],
    "P0_j": exp_data[solvent_data_key]["P0_j"],
    "C_f": exp_data["C_f"],
    "C_j": exp_data["C_j"],
    "def_t": max(exp_data[solid_data_key]["t"]),
    "rayon_initialisation": exp_data["R_s"] * 1.0e-9, # Le rayon est en nm -> conversion en m
    "name": f"{EXP_NAME_TO_RUN}_{CASE}",
    "R_vrai_m": exp_data["R_s"] * 1.0e-9,
}

# 5. Lancement de l'entraînement
print(f"\nLancement de l'entraînement du PINN pour {params['name']}...")
run(
    params_pinns=params_pinns,
    params=params,
    S_f=S_f,
    S_j=S_j,
    output_path=output_path,
    no_gui=True,
    no_interaction=True,
)
print("... Entraînement terminé.")

# 6. Post-traitement et affichage
# L'appel à affichage() est déjà à la fin de run(), donc les graphiques devraient déjà être générés.
# On peut le rappeler ici si on veut s'assurer que c'est fait ou si on le retire de run().
print(f"\nPost-traitement et affichage des résultats pour {params['name']}...")
try:
    affichage(output_path)
    print("... Graphiques générés avec succès.")
except Exception as e:
    print(f"[AVERTISSEMENT] Erreur lors de la génération des graphiques post-run : {e}")

print("\n=== FIN DE L'EXPÉRIENCE ===")