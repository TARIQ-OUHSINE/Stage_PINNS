# ==============================================================================
#           SCRIPT AUTONOME POUR LA GÉNÉRATION DE DONNÉES DE RÉFÉRENCE (FEM)
#
# Cette version intègre la classe DataGenerator originale et adapte la logique
# de pilotage pour assurer la compatibilité et la stabilité.
#
# Auteur: Tariq Ouhsine (adapté par l'IA)
# Date: 19/08/2025
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
    print("="*80); print("ERREUR CRITIQUE: FEniCSx non trouvé."); print(f"Détail: {e}"); exit()

# --- Dépendance PyTorch (requise par la classe DataGenerator) ---
try:
    import torch
except ImportError:
    print("="*80); print("ERREUR CRITIQUE: PyTorch non trouvé. Installez-le (pip install torch)"); exit()

# ==============================================================================
# SECTION 1: CLASSE DataGenerator ORIGINALE (INTÉGRÉE)
# ==============================================================================

# ==============================================================================
#           NOUVELLE CLASSE DataGenerator - VERSION ROBUSTE ET CORRIGÉE
# ==============================================================================

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
        # Le constructeur reste le même
        self.R = R; self.r_max = r_max; self.Tfinal = Tfinal; self.Nr = Nr; self.Nt = Nt
        self.dt = self.Tfinal / self.Nt
        self.tanh_slope = tanh_slope
        if tanh_slope == 0.:
            self.C = lambda r: ufl.conditional(ufl.lt(r, self.R), C_in, C_out)
            self.D = lambda r: ufl.conditional(ufl.lt(r, self.R), D_in, D_out)
            self.T1 = lambda r: ufl.conditional(ufl.lt(r, self.R), T1_in, T1_out)
            self.P0 = lambda r: ufl.conditional(ufl.lt(r, self.R), P0_in, P0_out)
        else:
            mid = 0.5 * (C_out + C_in); amp = 0.5 * (C_out - C_in)
            self.C = lambda r: mid + amp * ufl.tanh((r - self.R) / tanh_slope)
            mid = 0.5 * (D_out + D_in); amp = 0.5 * (D_out - D_in)
            self.D = lambda r: mid + amp * ufl.tanh((r - self.R) / tanh_slope)
            mid = 0.5 * (T1_out + T1_in); amp = 0.5 * (T1_out - T1_in)
            self.T1 = lambda r: mid + amp * ufl.tanh((r - self.R) / tanh_slope)
            mid = 0.5 * (P0_out + P0_in); amp = 0.5 * (P0_out - P0_in)
            self.P0 = lambda r: mid + amp * ufl.tanh((r - self.R) / tanh_slope)
        self.msh = None; self.V = None; self.P_time = None
        self.r_sorted = None; self.t_vec = None

def solve(self):
    # Maillage et espace
    self.msh = mesh.create_interval(MPI.COMM_WORLD, self.Nr, [0.0, self.r_max])
    self.V = fem.functionspace(self.msh, ("Lagrange", 1))

    # Coordonnée radiale et poids régularisé
    x = ufl.SpatialCoordinate(self.msh)[0]
    r = x
    h = self.r_max / self.Nr
    eps_w = ScalarType((0.5*h)**2)      # régularisation minimale du centre
    w = r**2 + eps_w

    # Coefficients (avec planchers de sûreté)
    C_expr_raw = self.C(r)
    D_expr_raw = self.D(r)
    T1_expr_raw = self.T1(r)
    P0_func = self.P0(r)

    # planchers numériques (max_value côté UFL)
    C_expr  = ufl.max_value(C_expr_raw, ScalarType(1.0))        # si jamais C→0
    D_expr  = ufl.max_value(D_expr_raw, ScalarType(1e-18))       # D très petit → plancher
    T1_expr = ufl.max_value(T1_expr_raw, ScalarType(1e-3))       # T1 très petit → plancher

    # Rampe sur P0
    t_now = fem.Constant(self.msh, ScalarType(0.0))
    T_ramp = max(self.Tfinal * 0.02, 10 * (self.Tfinal/self.Nt))
    P0_eff = ufl.conditional(t_now < T_ramp,
                             P0_func * (t_now / ScalarType(T_ramp)),
                             P0_func)

    # Inconnues et test
    P_n1 = ufl.TrialFunction(self.V)
    v    = ufl.TestFunction(self.V)
    P_n  = fem.Function(self.V)
    P_n.x.array[:] = 0.0

    dx = ufl.dx

    # Schéma implicite (Euler)
    dt = ScalarType(self.dt)
    mass_lhs = (C_expr/dt) * w * P_n1 * v * dx
    mass_rhs = (C_expr/dt) * w * P_n   * v * dx
    diff_lhs = (D_expr*C_expr) * w * ufl.dot(ufl.grad(P_n1), ufl.grad(v)) * dx
    reac_lhs = (C_expr/T1_expr) * w * P_n1 * v * dx
    reac_rhs = (C_expr/T1_expr) * w * P0_eff * v * dx

    a_form = mass_lhs + diff_lhs + reac_lhs
    L_form = mass_rhs + reac_rhs

    # Problème linéaire (tu peux garder LU si la régularisation est en place)
    problem = LinearProblem(
        a_form, L_form, [],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            # "mat_mumps_icntl_24": 1,  # si MUMPS dispo, tolérance pivot
        }
    )

    # Prépare l’ordonnancement global des dofs pour le rank 0
    n_loc = self.V.dofmap.index_map.size_local
    r_loc = self.V.tabulate_dof_coordinates()[:n_loc, 0]
    r_all = self.msh.comm.gather(r_loc, root=0)
    if self.msh.comm.rank == 0:
        r_glob = np.concatenate(r_all)
        order = np.argsort(r_glob)
        self.r_sorted = r_glob[order]
        P_hist = [np.zeros_like(self.r_sorted)]

    # Boucle en temps + rampe
    t = 0.0
    for _ in range(self.Nt):
        t += float(self.dt)
        t_now.value = ScalarType(t)

        P_new = problem.solve()

        # rassembler sur le rang 0
        P_loc = P_new.x.array[:n_loc]
        P_all = self.msh.comm.gather(P_loc, root=0)
        if self.msh.comm.rank == 0:
            P_glob = np.concatenate(P_all)[order]
            # garde une copie
            P_hist.append(P_glob.copy())

        # état suivant
        P_n.x.array[:] = P_new.x.array

    if self.msh.comm.rank == 0:
        self.P_time = np.array(P_hist)          # shape (Nt+1, Ndof)
        self.t_vec  = np.linspace(0.0, self.Tfinal, self.Nt+1)


# ==============================================================================
# SECTION 2: LOGIQUE DE PILOTAGE ADAPTÉE
# ==============================================================================

def main(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    base_output = Path(args.output_dir)
    data_file = Path(args.data_file)
    
    all_data = None
    if rank == 0:
        base_output.mkdir(exist_ok=True)
        print("="*60); print("Lancement des simulations de référence (FEM)"); print(f"Fichier de données : {data_file}"); print(f"Dossier de sortie  : {base_output}"); print("="*60)
        if not data_file.exists():
            print(f"ERREUR: Le fichier de données '{data_file}' est introuvable.")
        else:
            with open(data_file, "rb") as f:
                all_data = pickle.load(f)
            print(f"Trouvé {len(all_data)} cas à simuler dans le fichier .pkl.")
    all_data = comm.bcast(all_data, root=0)
    if all_data is None: return

    for case_name, exp_data in all_data.items():
        if rank == 0:
            print(f"\n--- Traitement du cas : {case_name} ---")

        try:
            solid_data_key, solvent_data_key = "CrisOn", "JuiceOn"
            C_ref, D_ref_nm2_s = 60.0, 500.0
            D_ref_m2_s = D_ref_nm2_s * 1e-18
            C_f = exp_data.get("C_f", C_ref)
            C_j = exp_data.get("C_j", C_ref)
            D_f_calculated = D_ref_m2_s * ((C_f / C_ref) ** (1/3))
            D_j_calculated = D_ref_m2_s * ((C_j / C_ref) ** (1/3))
            params = {
                "D_f": D_f_calculated, "D_j": D_j_calculated,
                "T_1_f": exp_data.get("T_1_f", 300.0), "T_1_j": exp_data[solid_data_key]["TB_j"],
                "P0_f": 1.0, "P0_j": exp_data[solvent_data_key]["P0_j"],
                "def_t": max(exp_data[solid_data_key]["t"]), "R_vrai_m": exp_data["R_s"] * 1.0e-9,
            }
            
            params_solver = {
                "R": params["R_vrai_m"], "r_max": params["R_vrai_m"] * 6.0,
                "C_in": C_f * 1000, "C_out": C_j * 1000,
                "D_in": params["D_f"], "D_out": params["D_j"],
                "T1_in": params["T_1_f"], "T1_out": params["T_1_j"],
                "P0_in": params["P0_f"], "P0_out": params["P0_j"],
                "Tfinal": params["def_t"], "Nr": 500, "Nt": 500,
                "tanh_slope": params["R_vrai_m"] * 0.05
            }
            
            if rank == 0: print("   Configuration et lancement de la simulation FEM...")
            solver = DataGenerator(**params_solver)
            solver.solve()
            
            # --- MODIFICATION DE LA LOGIQUE DE SAUVEGARDE ---
            if rank == 0:
                # On vérifie que la simulation a produit des résultats
                if solver.P_time is not None:
                    output_path = base_output / f"{case_name}_reference_FEM"
                    data_dir = output_path / "Data"
                    data_dir.mkdir(parents=True, exist_ok=True)
                    output_file = data_dir / "reference_FEM_solution.npz"
                    np.savez(
                        output_file,
                        P_rt=solver.P_time, # On accède directement à l'attribut
                        r=solver.r_sorted,    # On accède directement à l'attribut
                        t=solver.t_vec,       # On accède directement à l'attribut
                        params=params_solver
                    )
                    print(f"   -> Résultats sauvegardés dans {output_file}")
                else:
                    print(f"   *** ATTENTION: La simulation pour {case_name} n'a pas produit de résultats. ***")

        except Exception as e:
            if rank == 0:
                print(f"   *** ERREUR lors du traitement du cas {case_name}: {e} ***")
    
    if rank == 0: print("\n--- Toutes les simulations sont terminées. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancer les simulations de référence FEM.")
    parser.add_argument('--data_file', type=str, required=True, help="Chemin vers le fichier .pkl global.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier racine des résultats.")
    try:       
        args = parser.parse_args()
    except SystemExit:
        class Args: data_file = "donnees.pkl"; output_dir = "FEM_Results"
        args = Args()
    main(args)