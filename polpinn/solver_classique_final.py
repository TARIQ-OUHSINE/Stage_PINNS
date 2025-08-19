# ==============================================================================
#           SCRIPT AUTONOME POUR LA GÉNÉRATION DE DONNÉES DE RÉFÉRENCE (FEM)
#
# Version finale intégrant les corrections de conditionnement numérique.
# - Normalisation du poids géométrique r^2
# - Stabilisation du solveur linéaire (diagonal shift)
# - Planchers numériques et rampe sur P0
#
# Auteur: Tariq Ouhsine (adapté et corrigé par l'IA)
# Date: 20/08/2025
# ==============================================================================

import os
import numpy as np
import pickle
import argparse
from pathlib import Path
from collections.abc import Callable

try:
    from mpi4py import MPI
    import ufl
    from petsc4py.PETSc import ScalarType
    from dolfinx import mesh, fem
    from dolfinx.fem.petsc import LinearProblem
except ImportError as e:
    print("ERREUR: FEniCSx non trouvé. Assurez-vous d'être dans le bon environnement."); exit()

# ==============================================================================
# SECTION 1: CLASSE DataGenerator FINALE ET STABILISÉE
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
        self.R = R; self.r_max = r_max; self.Tfinal = Tfinal; self.Nr = Nr; self.Nt = Nt
        self.dt = self.Tfinal / self.Nt
        self.tanh_slope = tanh_slope
        
        # --- Garde-fou : Planchers numériques ---
        C_in = max(C_in, 1.0); C_out = max(C_out, 1.0)
        D_in = max(D_in, 1e-18); D_out = max(D_out, 1e-18)
        T1_in = max(T1_in, 1e-3); T1_out = max(T1_out, 1e-3)
        
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


    # ==============================================================================
    def solve(self):
        self.msh = mesh.create_interval(MPI.COMM_WORLD, self.Nr, [0.0, self.r_max])
        self.V = fem.functionspace(self.msh, ("Lagrange", 1))

        # --- Définition des variables et fonctions ---
        x = ufl.SpatialCoordinate(self.msh)[0]; r = x
        P_n1 = ufl.TrialFunction(self.V)  # Inconnue P au temps t_{n+1}
        v = ufl.TestFunction(self.V)     # Fonction test
        P_n = fem.Function(self.V)       # Solution connue au temps t_n
        P_n.x.array[:] = 0.0             # Condition initiale P(r, 0) = 0

        # --- Coefficients physiques ---
        C_expr = self.C(r)
        D_expr = self.D(r)
        T1_expr = self.T1(r)
        
        # --- Garde-fou 1 : Rampe sur la polarisation d'équilibre P0 ---
        # On fait monter P0 de 0 à sa valeur finale pour éviter un choc initial
        t_fencis = fem.Constant(self.msh, ScalarType(0.0))
        # La rampe dure 2% du temps total ou 10 pas de temps (on prend le plus long)
        T_ramp = max(self.Tfinal * 0.02, 10 * self.dt)
        P0_func = self.P0(r)
        P0_expr = ufl.conditional(t_fencis < T_ramp, P0_func * t_fencis / T_ramp, P0_func)
        
        
        # --- Formulation Faible (Variationnelle) ---
        # L'équation physique est :
        # C * dP/dt - Div(D*C*Grad(P)) = - C/T1 * (P - P0)
        #
        # Après discrétisation en temps (Euler implicite) :
        # C*(P_n1 - P_n)/dt - Div(D*C*Grad(P_n1)) = - C/T1 * (P_n1 - P0)
        #
        # On regroupe les termes inconnus (en P_n1) à gauche et les termes connus à droite :
        # C/dt*P_n1 - Div(D*C*Grad(P_n1)) + C/T1*P_n1 = C/dt*P_n + C/T1*P0
        #
        # Après multiplication par r^2*v, intégration et intégration par parties du terme de divergence :
        # Int[ C/dt*r^2*P_n1*v + D*C*r^2*Grad(P_n1)*Grad(v) + C/T1*r^2*P_n1*v ]dr = Int[ C/dt*r^2*P_n*v + C/T1*r^2*P0*v ]dr

        dx = ufl.dx
        
        # Côté gauche de l'équation (forme bilinéaire a_form, dépend de P_n1 et v)
        a_form = (C_expr * r**2 * P_n1 * v / self.dt) * dx \
               + (r**2 * D_expr * C_expr * ufl.dot(ufl.grad(P_n1), ufl.grad(v))) * dx \
               + (C_expr * r**2 / T1_expr * P_n1 * v) * dx

        # Côté droit de l'équation (forme linéaire L_form, dépend de P_n, P0 et v)
        L_form = (C_expr * r**2 * P_n * v / self.dt) * dx \
               + (C_expr * r**2 / T1_expr * P0_expr * v) * dx
               
        # --- Garde-fou 2 : Stabilisation du solveur linéaire (PETSc) ---
        # Évite les erreurs de pivot nul, surtout au premier pas de temps
        petsc_opts = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_shift_type": "NONZERO",
            "pc_factor_shift_amount": 1e-12,
        }

        problem = LinearProblem(a_form, L_form, [], petsc_options=petsc_opts)
        
        # --- Boucle temporelle ---
        if self.msh.comm.rank == 0:
            n_dofs = self.V.dofmap.index_map.size_global
            # Correction pour s'assurer que r_sorted a la bonne taille en parallèle
            if n_dofs == 0: # Si ce proc n'a pas de dofs, il doit recevoir les coords
                 coords = np.empty((0, self.msh.geometry.dim), dtype=np.float64)
                 self.r_sorted = self.msh.comm.bcast(coords, root=0) # Placeholder
            else:
                 self.r_sorted = self.V.tabulate_dof_coordinates()[:n_dofs, 0]

            P_hist = [np.zeros_like(self.r_sorted)] # On stocke l'état initial t=0

        time_current = 0.0
        for _ in range(self.Nt):
            # Mise à jour de la constante temps pour la rampe
            time_current += self.dt
            t_fencis.value = time_current
            
            P_new = problem.solve()
            
            # Seul le processus root (rang 0) stocke l'historique
            if self.msh.comm.rank == 0:
                P_hist.append(P_new.x.array.copy())

            # Mise à jour de la solution pour le pas de temps suivant
            P_n.x.array[:] = P_new.x.array
        
        if self.msh.comm.rank == 0:
            self.P_time = np.array(P_hist)
            self.t_vec = np.linspace(0, self.Tfinal, self.Nt + 1)

# ==============================================================================
# SECTION 2: LOGIQUE DE PILOTAGE
# ==============================================================================
def main(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    base_output = Path(args.output_dir)
    data_file = Path(args.data_file)
    all_data = None
    if rank == 0:
        base_output.mkdir(exist_ok=True)
        print("="*60); print("Lancement des simulations de référence (FEM) - Version Stabilisée"); print(f"Fichier de données : {data_file}"); print(f"Dossier de sortie  : {base_output}"); print("="*60)
        if not data_file.exists(): print(f"ERREUR: Le fichier de données '{data_file}' est introuvable.")
        else:
            with open(data_file, "rb") as f: all_data = pickle.load(f)
            print(f"Trouvé {len(all_data)} cas à simuler.")
    all_data = comm.bcast(all_data, root=0)
    if all_data is None: return

    for case_name, exp_data in all_data.items():
        if rank == 0: print(f"\n--- Traitement du cas : {case_name} ---")
        try:
            solid_data_key, solvent_data_key = "CrisOn", "JuiceOn"
            C_ref, D_ref_nm2_s = 60.0, 500.0; D_ref_m2_s = D_ref_nm2_s * 1e-18
            C_f = exp_data.get("C_f", C_ref); C_j = exp_data.get("C_j", C_ref)
            D_f_calculated = D_ref_m2_s * ((C_f / C_ref) ** (1/3)); D_j_calculated = D_ref_m2_s * ((C_j / C_ref) ** (1/3))
            params = {"D_f": D_f_calculated, "D_j": D_j_calculated, "T_1_f": exp_data.get("T_1_f", 300.0), "T_1_j": exp_data[solid_data_key]["TB_j"], "P0_f": 1.0, "P0_j": exp_data[solvent_data_key]["P0_j"], "def_t": max(exp_data[solid_data_key]["t"]), "R_vrai_m": exp_data["R_s"] * 1.0e-9}
            
            params_solver = {"R": params["R_vrai_m"], "r_max": params["R_vrai_m"] * 6.0, "C_in": C_f * 1000, "C_out": C_j * 1000, "D_in": params["D_f"], "D_out": params["D_j"], "T1_in": params["T_1_f"], "T1_out": params["T_1_j"], "P0_in": params["P0_f"], "P0_out": params["P0_j"], "Tfinal": params["def_t"], "Nr": 1000, "Nt": 1000, "tanh_slope": params["R_vrai_m"] * 0.05}
            
            if rank == 0: print("   Configuration et lancement de la simulation...")
            solver = DataGenerator(**params_solver)
            solver.solve()
            
            if rank == 0:
                if solver.P_time is not None:
                    output_path = base_output / f"{case_name}_reference_FEM"; data_dir = output_path / "Data"; data_dir.mkdir(parents=True, exist_ok=True); output_file = data_dir / "reference_FEM_solution.npz"
                    np.savez(output_file, P_rt=solver.P_time, r=solver.r_sorted, t=solver.t_vec, params=params_solver)
                    print(f"   -> Résultats sauvegardés dans {output_file}")
                else: print(f"   *** ATTENTION: La simulation pour {case_name} n'a pas produit de résultats. ***")
        except Exception as e:
            if rank == 0: print(f"   *** ERREUR lors du traitement du cas {case_name}: {e} ***")
    
    if rank == 0: print("\n--- Toutes les simulations sont terminées. ---")

# ==============================================================================
#                 BLOC D'EXÉCUTION PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # On définit les arguments attendus en ligne de commande
    parser = argparse.ArgumentParser(description="Lancer les simulations de référence FEM.")
    parser.add_argument('--data_file', type=str, required=True, 
                        help="Chemin vers le fichier de données .pkl global.")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Chemin vers le dossier racine où stocker les résultats.")
    
    # On parse les arguments fournis par la ligne de commande (via Slurm)
    args = parser.parse_args()
    
    # On appelle la fonction principale avec les arguments parsés
    main(args)