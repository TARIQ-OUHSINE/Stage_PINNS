import pandas as pd
import pickle
import os

def read_comsol_txt(filepath):
    # Lire en ignorant les lignes qui commencent par %
    df = pd.read_csv(filepath, sep=r"\s+", comment='%', header=None, skip_blank_lines=True)

    # Vérification rapide
    if df.shape[1] < 4:
        raise ValueError("Le fichier ne contient pas les 4 colonnes attendues.")

    # Création du dictionnaire de sortie
    data = {
        "P0_j": float(df.iloc[0, 0]),
        "TB_j": float(df.iloc[0, 1]),
        "t": df.iloc[:, 2].tolist(),
        "P_moy": df.iloc[:, 3].tolist()
    }
    return data

def save_dict_to_pkl(data, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Données enregistrées dans : {output_path}")

if __name__ == "__main__":
    # 📂 Chemin vers le fichier à lire
    filepath = r"C:\Users\TARIQ\Downloads\partage_tariq - Copie\stage-l2-2024-ve\stage-l2-2024-ve\FolderRun\FolderRun\110_20_300_50\CristalOff.txt"

    # 📥 Lecture
    data = read_comsol_txt(filepath)

    # 💾 Enregistrement
    save_path = os.path.join(os.path.dirname(filepath), "CristalOff.pkl")
    save_dict_to_pkl(data, save_path)
