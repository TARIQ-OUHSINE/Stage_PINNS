import json
import shutil
import torch
from pathlib import Path
from polpinn.PINN import run
from polpinn.utils import get_output_dir, get_data_dir

# === CONFIGURATION ===
R_inits = [ 100e-9 ]               # 100e-9, 150e-9, 200e-9, , 300e-9, 350e-9, 400e-9, 450e-9, 500e-9, 550e-9, 600e-9, 650e-9, 700e-9, 750e-9, 800e-9, 850e-9, 900e-9, 950e-9,1000e-9
data_path = get_data_dir() / "Moyenne_homo_HyperP"
base_output = get_output_dir() / "multiple_runs"

params_template = {
    "D_f": 5e-16,
    "D_j": 5e-16,
    "T_1": 20,
    "T_B": 3,
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
    "lr_R": 0.0005,
    "epoch": 10000,
    "var_R": True,
}

# === RUN MULTIPLE TESTS ===
results = []

for R in R_inits:
    print(f"\n=== Running for R_init = {R * 1e9:.1f} nm ===")
    params = params_template.copy()
    params["rayon_initialisation"] = R
    params["name"] = f"Sim_10K_Rv_R_bis_0_PINN_AdamW_Axe1_{int(R * 1e9)}nm"
    output_path = base_output / params["name"]

    # remove existing directory to re-run cleanly
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

    # Load the loss file
    loss_path = output_path / "Data" / "loss.json"
    params_json_path = output_path / "Data" / "params.json"
    if loss_path.exists() and params_json_path.exists():
        with open(loss_path, "r") as f:
            losses = json.load(f)
            min_loss = min(losses[0])
            final_R = None
            if params_pinns["var_R"]:
                final_R = losses[-1][-1]

        with open(params_json_path, "r") as f:
            pjson = json.load(f)
            ordre_R = pjson["ordre_R"]

        results.append({
            "R_init_nm": R * 1e9,
            "min_loss": min_loss,
            "final_R_nm": final_R * 10 ** (ordre_R + 1) * 1e9 if final_R else R * 1e9
        })



# === SAVE RESULTS ===
summary_file = base_output / "summary_results_Rv_Axe1.json"

# Charger les anciens résultats s'ils existent
if summary_file.exists():
    with open(summary_file, "r") as f:
        existing_results = json.load(f)
else:
    existing_results = []

# Ajouter les nouveaux résultats
existing_results.extend(results)

# Enregistrer le tout
with open(summary_file, "w") as f:
    json.dump(existing_results, f, indent=2)

# Affichage console
print("\n=== Résultats ===")
for res in results:
    print(f"R_init: {res['R_init_nm']:.1f} nm | Min loss: {res['min_loss']:.2e} | Final R: {res['final_R_nm']:.1f} nm")
