import json
import numpy as np
import matplotlib.pyplot as plt
from polpinn.utils import get_output_dir

# === Chemin vers ton fichier JSON ===
base_output = get_output_dir() / "multiple_runs"
json_path = base_output/"summary_results_4.json"                       


# === Lecture du fichier JSON ===
with open(json_path, "r") as f:
    data = json.load(f)

# === Extraction des valeurs ===
R_values = [entry["R_init_nm"] for entry in data]
min_losses = [entry["min_loss"] for entry in data]

# === Conversion en array pour tri
R_values = np.array(R_values)
min_losses = np.array(min_losses)

# === Tri selon R croissant
sorted_indices = np.argsort(R_values)
R_sorted = R_values[sorted_indices]
loss_sorted = min_losses[sorted_indices]

# === Tracé ===
plt.figure(figsize=(8, 5))
plt.plot(R_sorted, loss_sorted, marker='o')

# Échelle logarithmique pour la loss
plt.yscale('log')
plt.ylim(1e-7, 1e-3)

# Pas de 50 pour R
plt.xticks(np.arange(min(R_sorted), max(R_sorted)+1, 50))

plt.xlabel("R_init (nm)")
plt.ylabel("Loss minimale (log scale)")
plt.title("Loss minimale en fonction de R_init (échelle log)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
