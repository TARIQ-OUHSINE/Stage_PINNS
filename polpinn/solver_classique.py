# ==============================================================================
#           SCRIPT AUTONOME POUR LA GÉNÉRATION DE DONNÉES DE RÉFÉRENCE (FEM)
#
# Ce script est conçu pour fonctionner en tandem avec le script PINN.
# - Il lit les paramètres depuis le même fichier de données .pkl.
# - Il utilise la même logique de calcul des paramètres.
# - Il stocke les résultats dans une arborescence de dossiers similaire.
#
# Auteur: Tariq Ouhsine (adapté par l'IA)
# Date: 19/08/2025
#
# Pré-requis: FEniCSx (dolfinx), petsc4py, mpi4py, numpy, pickle
#
# Lancement sur un cluster (via un script Slurm):
#   srun python run_fem_simulations.py --data_file <chemin_pkl> --output_dir <dossier_sortie>
#
# Exemple de commande dans un script Slurm:
#   srun python run_fem_simulations.py --data_file donnees.pkl --output_dir ./FEM_Results
#
# ==============================================================================

import os
import numpy as np
import pickle
import argparse
from pathlib import Path
from collections.abc import Callable

# --- Dépendances FEniCSx / MPI ---
try:
    from mpi4py import MPI
    import ufl
    from petsc4py.PETSc import ScalarType
    from dolfinx import mesh, fem
    from dolfinx.fem.petsc import LinearProblem
except ImportError as e:
    print("=" * 80)
    print("ERREUR CRITIQUE: FEniCSx (dolfinx) ou une de ses dépendances n'est pas installée.")
    print(f"Détail: {e}")
    print("\nSur un cluster, assurez-vous d'avoir chargé le bon module ou activé le bon environnement Conda.")
    print("=" * 80)
    exit()

# ==============================================================================
# SECTION 1: LE SOLVEUR NUMÉRIQUE (CLASSE DataGenerator - Inchangée)
# ==============================================================================
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import torch
from torch import Tensor
from collections.abc import Callable
import os


class DataGenerator:
    def __init__(self,
                 R: float, r_max: float,
                 C_in: float, C_out: float,
                 D_in: float, D_out: float,
                 T1_in: float, T1_out: float,
                 P0_in: float, P0_out: float,
                 Tfinal: float = 10.,
                 Nr: int = 100,
                 Nt: int = 100,
                 tanh_slope: float = 0.,
                 ):

        self.R = R
        self.r_max = r_max
        self.Tfinal = Tfinal
        self.Nr = Nr
        self.Nt = Nt
        self.dt = self.Tfinal / self.Nt
        self.tanh_slope = tanh_slope

        self.C_in = C_in
        self.C_out = C_out
        self.D_in = D_in
        self.D_out = D_out
        self.T1_in = T1_in
        self.T1_out = T1_out
        self.P0_in = P0_in
        self.P0_out = P0_out

        if tanh_slope == 0.:
            self.C: Callable = lambda r: ufl.conditional(ufl.lt(r, self.R), C_in, C_out)
            self.D: Callable = lambda r: ufl.conditional(ufl.lt(r, self.R), D_in, D_out)
            self.T1: Callable = lambda r: ufl.conditional(ufl.lt(r, self.R), T1_in, T1_out)
            self.P0: Callable = lambda r: ufl.conditional(ufl.lt(r, self.R), P0_in, P0_out)
        else:
            mid = 0.5 * (C_out + C_in)
            amp = 0.5 * (C_out - C_in)
            self.C = lambda r: mid + amp * ufl.tanh((r - self.R) / tanh_slope)
            mid = 0.5 * (D_out + D_in)
            amp = 0.5 * (D_out - D_in)
            self.D = lambda r: mid + amp * ufl.tanh((r - self.R) / tanh_slope)
            mid = 0.5 * (T1_out + T1_in)
            amp = 0.5 * (T1_out - T1_in)
            self.T1 = lambda r: mid + amp * ufl.tanh((r - self.R) / tanh_slope)
            mid = 0.5 * (P0_out + P0_in)
            amp = 0.5 * (P0_out - P0_in)
            self.P0 = lambda r: mid + amp * ufl.tanh((r - self.R) / tanh_slope)

        self.msh = None
        self.V = None
        self.P_time = []
        self.r_sorted = None
        self.t_vec = None

    def solve(self):
        self.msh = mesh.create_interval(MPI.COMM_WORLD, self.Nr, [0.0, self.r_max])
        self.V = fem.functionspace(self.msh, ("Lagrange", 1))
        x = ufl.SpatialCoordinate(self.msh)[0]
        r = x
        w = r**2
        C_expr = self.C(r)
        D_expr = self.D(r)
        T1_expr = self.T1(r)
        P0_expr = self.P0(r)
        P_n1 = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        P_n = fem.Function(self.V)
        P_n.x.array[:] = 0.0
        self.P_time.append(P_n)
        dx = ufl.dx
        mass_lhs = (C_expr / ScalarType(self.dt)) * w * P_n1 * v * dx
        mass_rhs = (C_expr / ScalarType(self.dt)) * w * P_n * v * dx
        diff_lhs = D_expr * C_expr * w * ufl.dot(ufl.grad(P_n1), ufl.grad(v)) * dx
        react_lhs = (C_expr / T1_expr) * w * P_n1 * v * dx
        react_rhs = (C_expr / T1_expr) * w * P0_expr * v * dx
        a_form = mass_lhs + diff_lhs + react_lhs
        L_form = mass_rhs + react_rhs
        problem = LinearProblem(
            a_form, L_form, [],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        n_loc = self.V.dofmap.index_map.size_local
        r_loc = self.V.tabulate_dof_coordinates()[:n_loc, 0]
        r_all = self.msh.comm.gather(r_loc, root=0)
        if self.msh.comm.rank == 0:
            r_glob = np.concatenate(r_all)
            order = np.argsort(r_glob)
            self.r_sorted = r_glob[order]
            P_hist = []
        t = 0.0
        for _ in range(self.Nt):
            t += self.dt
            P_new = problem.solve()
            P_loc = P_new.x.array[:n_loc]
            P_all = self.msh.comm.gather(P_loc, root=0)
            if self.msh.comm.rank == 0:
                P_glob = np.concatenate(P_all)
                P_hist.append(P_glob[order].copy())
            P_n.x.array[:] = P_new.x.array
        if self.msh.comm.rank == 0:
            self.P_time = np.array(P_hist)
            self.t_vec = np.linspace(self.dt, self.Tfinal, self.Nt)

    def plot(self, dir: str = "data/plot"):
        if self.msh.comm.rank == 0:
            if self.P_time is None or self.r_sorted is None or self.t_vec is None:
                return
            import matplotlib.pyplot as plt
            os.makedirs(dir, exist_ok=True)
            plt.figure(figsize=(8, 5))
            plt.imshow(self.P_time,
                       extent=[0.0, self.r_max, 0.0, self.Tfinal],
                       origin="lower",
                       aspect="auto",
                       cmap="viridis")
            plt.colorbar(label=r"$P(r,t)$")
            plt.xlabel(r"$r$ (m)")
            plt.ylabel(r"$t$ (s)")
            plt.title("Évolution de la polarisation P(r,t)")
            plt.tight_layout()
            filename = f"{dir}/{self.R}_{self.C_in}_{self.C_out}_{self.D_in}_{self.D_out}_{self.P0_in}_{self.P0_out}_{self.T1_in}_{self.T1_out}.png"
            plt.savefig(filename, dpi=180)


    # REMPLACEZ L'ANCIENNE MÉTHODE 'get' PAR CELLE-CI :
    def get(self) -> dict:
        """
        Retourne les résultats de la simulation sous forme de dictionnaire simple.
        Cette fonction est conçue pour être appelée en parallèle : seul le
        processeur de rang 0 renverra des données.
        """
        if self.msh.comm.rank == 0:
            # Vérifie que la simulation a bien tourné
            if self.P_time is None or self.r_sorted is None or self.t_vec is None:
                return {}
                
            # On retourne un dictionnaire propre avec les bonnes clés
            to_send = {
                "P_rt": self.P_time, # La matrice de solution
                "r": self.r_sorted,  # Le vecteur des rayons
                "t": self.t_vec      # Le vecteur des temps
            }
            return to_send
        return {}

# ==============================================================================
# SECTION 2: LOGIQUE DE PILOTAGE DES SIMULATIONS (Adaptée au workflow PINN)
# ==============================================================================

def main(args):
    """
    Fonction principale qui orchestre la simulation de tous les cas.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    base_output = Path(args.output_dir)
    data_file = Path(args.data_file)
    
    all_data = None
    if rank == 0:
        base_output.mkdir(exist_ok=True)
        print("="*60)
        print("Lancement des simulations de référence (FEM)")
        print(f"Fichier de données : {data_file}")
        print(f"Dossier de sortie  : {base_output}")
        print("="*60)

        if not data_file.exists():
            print(f"ERREUR: Le fichier de données '{data_file}' est introuvable.")
        else:
            with open(data_file, "rb") as f:
                all_data = pickle.load(f)
            print(f"Trouvé {len(all_data)} cas à simuler dans le fichier .pkl.")

    # Le processus principal diffuse les données à tous les autres processus
    all_data = comm.bcast(all_data, root=0)
    if all_data is None:
        return # Arrête tout si le fichier de données n'a pas pu être lu

    # --- Boucle sur chaque cas ---
    for case_name, exp_data in all_data.items():
        if rank == 0:
            print(f"\n--- Traitement du cas : {case_name} ---")

        try:
            # 1. Extraction des paramètres (logique identique au script PINN)
            solid_data_key, solvent_data_key = "CrisOn", "JuiceOn"

            C_ref, D_ref_nm2_s = 60.0, 500.0
            D_ref_m2_s = D_ref_nm2_s * 1e-18
            
            # Concentrations en mol/L
            C_f = exp_data.get("C_f", C_ref)
            C_j = exp_data.get("C_j", C_ref)

            # Calcul des coefficients de diffusion en m^2/s
            D_f_calculated = D_ref_m2_s * ((C_f / C_ref) ** (1/3))
            D_j_calculated = D_ref_m2_s * ((C_j / C_ref) ** (1/3))

            # Assemblage de tous les paramètres
            params = {
                "D_f": D_f_calculated, 
                "D_j": D_j_calculated,
                "T_1_f": exp_data.get("T_1_f", 300.0), 
                "T_1_j": exp_data[solid_data_key]["TB_j"],
                "P0_f": 1.0, 
                "P0_j": exp_data[solvent_data_key]["P0_j"],
                "def_t": max(exp_data[solid_data_key]["t"]),
                "R_vrai_m": exp_data["R_s"] * 1.0e-9,
            }
            
            # 2. Définition des paramètres pour le solveur
            params_solver = {
                "R": params["R_vrai_m"],
                "r_max": params["R_vrai_m"] * 6.0,
                "C_in": C_f * 1000,   # Cristal, conversion mol/L -> mol/m^3
                "C_out": C_j * 1000,  # Jus, conversion mol/L -> mol/m^3
                "D_in": params["D_f"],
                "D_out": params["D_j"],
                "T1_in": params["T_1_f"],
                "T1_out": params["T_1_j"],
                "P0_in": params["P0_f"],
                "P0_out": params["P0_j"],
                "Tfinal": 0.1,
                "Nr": 500,          # Résolution spatiale (r)
                "Nt": 2000,          # Résolution temporelle (t)
                "tanh_slope": params["R_vrai_m"] * 0.05
            }

            # 3. Lancement du solveur
            if rank == 0: print("   Configuration et lancement de la simulation FEM...")
            solver = DataGenerator(**params_solver)
            solver.solve()
            solver.plot()
            
            # 4. Récupération et sauvegarde des résultats
            results = solver.get()
            
            if rank == 0 and results:
                # Création de la structure de dossier (identique au PINN)
                output_path = base_output / f"{case_name}_reference_FEM"
                data_dir = output_path / "Data"
                data_dir.mkdir(parents=True, exist_ok=True)
                
                # Sauvegarde dans un fichier .npz pour tout regrouper
                output_file = data_dir / "reference_FEM_solution.npz"
                np.savez(
                    output_file,
                    P_rt=results["P_rt"],
                    r=results["r"],
                    t=results["t"],
                    params=params_solver
                )
                print(f"   -> Résultats sauvegardés dans {output_file}")
            
        except Exception as e:
            if rank == 0:
                print(f"   *** ERREUR lors du traitement du cas {case_name}: {e} ***")
    
    if rank == 0:
        print("\n--- Toutes les simulations sont terminées. ---")

# ==============================================================================
# POINT D'ENTRÉE DU SCRIPT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancer les simulations de référence FEM pour les cas d'un fichier .pkl.")
    parser.add_argument('--data_file', type=str, required=True, help="Chemin vers le fichier de données .pkl global.")
    parser.add_argument('--output_dir', type=str, required=True, help="Chemin vers le dossier racine où stocker les résultats.")
    
    # Bloc pour permettre des tests faciles sans ligne de commande (comme dans le script PINN)
    try:       
        args = parser.parse_args()
    except SystemExit:
        # Valeurs par défaut si le script est lancé sans arguments (ex: dans un IDE)
        print("INFO: Aucun argument de ligne de commande. Utilisation des valeurs par défaut.")
        class Args:
            data_file = "donnees.pkl"
            output_dir = "FEM_Results"
        args = Args()
    
    main(args)