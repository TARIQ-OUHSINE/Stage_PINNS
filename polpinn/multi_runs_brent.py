import json
import shutil
import torch
from pathlib import Path
from scipy.optimize import minimize_scalar
from polpinn.PINN import run
from polpinn.utils import get_output_dir, get_data_dir

# === CONFIGURATION ===
data_path = get_data_dir() / "Moyenne_homo_HyperP"
base_output = get_output_dir() / "multiple_runs"

params_template = {
    "D_f": 5e-16,
    "D_j": 5e-16,
    "T_1": 20,
    "T_B": 1,
    "P0_f": 1,
    "P0_j": 200,
    "R_bis": 0.0,
    "def_t": 20,
    "name": "Sim",  
    "rayon_initialisation": None 
}

params_pinns = {
    "nb_hidden_layer": 2,
    "nb_hidden_perceptron": 32,
    "lr": 0.001,
    "lr_R": 0.0001,
    "epoch": 100000,
    "var_R": False,
}

results = []
call_count = {"value": 0}
max_calls = 20

def objective(R):
    if call_count["value"] >= max_calls:
        print(f"Maximum number of evaluations ({max_calls}) reached.")
        return float("inf")
    
    call_count["value"] += 1
    R_nm = R * 1e9
    print(f"\n[{call_count['value']}/{max_calls}] Running for R_init = {R_nm:.1f} nm")

    # Set up simulation
    params = params_template.copy()
    params["rayon_initialisation"] = R
    params["name"] = f"BrentSearch_R_100K_xatol_9_{int(R_nm)}nm"
    output_path = base_output / params["name"]

    if output_path.exists():
        shutil.rmtree(output_path)

    run(
        params_pinns=params_pinns,
        params=params,
        data_path=data_path,
        output_path=output_path,
        no_gui=True,
        no_interaction=True,
    )

    # Load loss
    loss_path = output_path / "Data" / "loss.json"
    if loss_path.exists():
        with open(loss_path, "r") as f:
            losses = json.load(f)
            min_loss = min(losses[0])
            final_R = losses[-1][-1] if params_pinns["var_R"] else R
            results.append({
                "R_init_nm": R_nm,
                "min_loss": min_loss,
                "final_R_nm": final_R * 1e9
            })
            return min_loss
    else:
        print(f"Fichier loss.json manquant pour R = {R_nm:.1f} nm")
        return float("inf")

# === Lancer Brent avec bornes ===
res = minimize_scalar(
    objective,
    bounds=(100e-9, 1000e-9),
    method='bounded',
    options={
        "xatol": 1e-9  # tolérance très fine (on limite manuellement par call_count)
    }
)

# === Enregistrer les résultats ===
summary_file = base_output / "summary_results_4.json"

# Charger anciens résultats
if summary_file.exists():
    with open(summary_file, "r") as f:
        existing_results = json.load(f)
else:
    existing_results = []

# Ajouter en évitant les doublons
for new in results:
    existing_results = [r for r in existing_results if r["R_init_nm"] != new["R_init_nm"]]
    existing_results.append(new)

# Trier
existing_results = sorted(existing_results, key=lambda r: r["R_init_nm"])

with open(summary_file, "w") as f:
    json.dump(existing_results, f, indent=2)

# === Affichage final ===
print("\n=== Résultats ===")
for resu in results:
    print(f"R_init: {resu['R_init_nm']:.1f} nm | Min loss: {resu['min_loss']:.2e} | Final R: {resu['final_R_nm']:.1f} nm")

if results:
    best = min(results, key=lambda r: r["min_loss"])
    print(f"\n R optimal trouvé : {best['R_init_nm']:.1f} nm | Loss: {best['min_loss']:.2e}")
else:
    print("\n Aucun résultat n'a été généré.")
