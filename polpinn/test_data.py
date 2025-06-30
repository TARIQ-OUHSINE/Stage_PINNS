import os
import pickle
from pprint import pprint

def load_saved_data():
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(root_path, "data_1", "donnees.pkl")
    print(f"Chargement depuis : {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier non trouvé : {data_path}")
    
    with open(data_path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    all_data = load_saved_data()
    print(f"{len(all_data)} dossiers chargés.\n")

    folder_name = "11_58_300_5000"

    if folder_name in all_data:
        print(f"Données du dossier '{folder_name}' :\n")
        
        pprint(all_data[folder_name], depth=2)
        
        """data = all_data[folder_name]["CrisOff"]
        
        print(f"P0_j : {data['P0_j']:.17f}")
        print(f"TB_j : {data['TB_j']:.17f}")
        print("t (premiers points) :")
        for t in data["t"][:5]:
            print(f"  {t:.17f}")
        print("P_moy (premiers points) :")
        for p in data["P_moy"][:5]:
            print(f"  {p:.17f}")   """

    else:
        print(f"Dossier '{folder_name}' introuvable dans les données.")
