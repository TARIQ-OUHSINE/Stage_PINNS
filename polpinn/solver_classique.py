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

    # --- LA MÉTHODE SOLVE() ENTIÈREMENT RÉÉCRITE ET CORRIGÉE ---
    def solve(self):
        self.msh = mesh.create_interval(MPI.COMM_WORLD, self.Nr, [0.0, self.r_max])
        self.V = fem.functionspace(self.msh, ("Lagrange", 1))

        x = ufl.SpatialCoordinate(self.msh)[0]; r = x
        P_n1 = ufl.TrialFunction(self.V); v = ufl.TestFunction(self.V)
        P_n = fem.Function(self.V); P_n.x.array[:] = 0.0

        C_expr = self.C(r); D_expr = self.D(r); T1_expr = self.T1(r)
        
        # --- TECHNIQUE DE STABILISATION (Rampe sur P0) ---
        t_fencis = fem.Constant(self.msh, ScalarType(0.0))
        # Rampe sur les premiers 2% du temps total ou 10*dt, la plus grande des deux
        T_ramp = max(self.Tfinal * 0.02, 10 * self.dt)
        P0_func = self.P0(r)
        P0_expr = ufl.conditional(t_fencis < T_ramp, P0_func * t_fencis / T_ramp, P0_func)
        
        # --- FORMULATION FAIBLE MATHÉMATIQUEMENT CORRECTE ---
        # Équation: C * dP/dt = (1/r^2)*d/dr(r^2*D*C*dP/dr) - C*(P-P0)/T1
        # Multipliée par r^2*v et intégrée :
        # Integral[ C*r^2*dP/dt*v + r^2*D*C*dP/dr*dv/dr + C*r^2/T1*(P-P0)*v ] dr = 0
        dx = ufl.dx
        
        # Termes en P_n1 (futur) sur le côté gauche (a_form)
        a_form = (C_expr * r**2 * P_n1 * v / self.dt) * dx \
               + (r**2 * D_expr * C_expr * ufl.dot(ufl.grad(P_n1), ufl.grad(v))) * dx \
               + (C_expr * r**2 / T1_expr * P_n1 * v) * dx

        # Termes connus (P_n et P0) sur le côté droit (L_form)
        L_form = (C_expr * r**2 * P_n * v / self.dt) * dx \
               + (C_expr * r**2 / T1_expr * P0_expr * v) * dx

        # --- Boucle temporelle ---
        problem = LinearProblem(a_form, L_form, [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        
        if self.msh.comm.rank == 0:
            n_dofs = self.V.dofmap.index_map.size_global
            self.r_sorted = self.V.tabulate_dof_coordinates()[:n_dofs, 0]
            P_hist = [np.zeros_like(self.r_sorted)] # t=0

        time_current = 0.0
        for _ in range(self.Nt):
            time_current += self.dt
            t_fencis.value = time_current
            
            P_new = problem.solve()
            
            if self.msh.comm.rank == 0:
                P_hist.append(P_new.x.array.copy())
            P_n.x.array[:] = P_new.x.array
        
        if self.msh.comm.rank == 0:
            self.P_time = np.array(P_hist)
            self.t_vec = np.linspace(0, self.Tfinal, self.Nt + 1)

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