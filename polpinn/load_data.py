import os
import pickle
import pandas as pd

def parse_folder_name(folder_name):
    parts = folder_name.split('_')
    if len(parts) < 4:
        raise ValueError(f"Nom de dossier trop court pour extraire les paramètres : {folder_name}")
    try:
        C_j, C_f, T_1, R_s = map(float, parts[-4:])
        return C_j, C_f, T_1, R_s
    except Exception as e:
        raise ValueError(f"Impossible d’extraire les paramètres depuis : {folder_name}") from e


def read_file(filepath):
    df = pd.read_csv(filepath, delim_whitespace=True, skiprows=5, header=None)
    if df.shape[1] < 4:
        raise ValueError(f"Fichier {filepath} ne contient pas au moins 4 colonnes.")
    
    return {
        "P0_j": float(df.iloc[0, 0]),
        "TB_j": float(df.iloc[0, 1]),
        "t": df.iloc[:, 2].tolist(),
        "P_moy": df.iloc[:, 3].tolist()
    }

def load_all_data(folder_run_path):
    all_data = {}

    print(f"Chemin vers FolderRun : {folder_run_path}")

    for folder_name in os.listdir(folder_run_path):
        folder_path = os.path.join(folder_run_path, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        try:
            C_j, C_f, T_1, R_s = parse_folder_name(folder_name)
        except ValueError:
            print(f"Nom de dossier ignoré : {folder_name}")
            continue

        data_entry = {
            "C_j": C_j,
            "C_f": C_f,
            "T_1": T_1,
            "R_s": R_s
        }

        try:
            data_entry["CrisOff"] = read_file(os.path.join(folder_path, "CristalOff.txt"))
            data_entry["CrisOn"]  = read_file(os.path.join(folder_path, "CristalOn.txt"))
            data_entry["JuiceOff"] = read_file(os.path.join(folder_path, "JuiceOff.txt"))
            data_entry["JuiceOn"]  = read_file(os.path.join(folder_path, "JuiceOn.txt"))
        except Exception as e:
            print(f"Erreur de lecture dans {folder_name}: {e}")
            continue

        all_data[folder_name] = data_entry

    return all_data


def save_data(data, project_root):

    output_dir = os.path.join(project_root, "data_1")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "donnees.pkl")
    
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Données sauvegardées dans : {output_path}")



if __name__ == "__main__":

    root_project = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    folder_run_path = os.path.join(root_project, "FolderRun", "FolderRun")
    data = load_all_data(folder_run_path)
    save_data(data, root_project)
