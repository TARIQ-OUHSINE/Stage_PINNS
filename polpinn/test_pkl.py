import pickle
import os

def load_dict_from_pkl(pkl_path):
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Fichier introuvable : {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

if __name__ == "__main__":
    # Chemin du fichier .pkl 
    pkl_path = r"C:\Users\TARIQ\Downloads\partage_tariq - Copie\stage-l2-2024-ve\stage-l2-2024-ve\FolderRun\FolderRun\110_20_300_50\CristalOff.pkl"

    # 📤 Chargement des données
    data = load_dict_from_pkl(pkl_path)

    # 🖨️ Affichage
    print(" Données chargées :")
    print("P0_j =", data["P0_j"])
    print("TB_j =", data["TB_j"])
    print("Nombre de points =", len(data["t"]))
    print("\nExtrait (t, P_moy) :")
    for t, p in zip(data["t"][:40], data["P_moy"][:40]): 
        print(f"{t} s → {p}")
