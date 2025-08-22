#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solver_classique_final.py
Lit un pickle { case_name: exp_data }, calcule les paramètres FEM attendus,
résout tous les cas (FEM 1D radial robuste, GPU-ready via PETSc), et enregistre :
  - <output_dir>/<case_name>_reference_FEM/Data/reference_FEM_solution.npz
  - <output_dir>/recap_polarisation.csv  (min/max + moyennes sphériques finales)

À lancer (exemples) :
  python solver_classique_final.py --data_file ../data_test/donnees.pkl --output_dir ../FEM_Results
  srun python solver_classique_final.py --data_file /chemin/donnees.pkl --output_dir /chemin/FEM_Results

Dépendances : mpi4py, numpy, ufl, petsc4py, dolfinx
"""
from __future__ import annotations
import os, sys, csv, time, argparse, pickle
from pathlib import Path
import numpy as np

# --- FEniCSx / PETSc ---
try:
    from mpi4py import MPI
    import ufl
    from petsc4py.PETSc import ScalarType
    from dolfinx import mesh, fem
    from dolfinx.fem.petsc import LinearProblem
except Exception as e:
    print("ERREUR: FEniCSx/PETSc non disponible dans cet environnement :", e)
    sys.exit(1)


# ============================================================================
#                         CLASSE FEM ROBUSTE
# ============================================================================
class DataGenerator:
    """
    Équation radiale 1D :
      C(r) dP/dt = (1/r^2) d/dr [ r^2 D(r) C(r) dP/dr ] - C(r) (P - P0(r)) / T1(r)

    Discrétisation :
      - FEM P1 sur [0, r_max]
      - Poids régularisé & NORMALISÉ : w = (r^2 + eps) / r_max^2
      - Euler implicite en temps
      - Stabilisations : planchers numériques (C,D,T1), rampe P0, diagonal shift
      - Solveur LU avec NONZERO pivot shift (ou itératif si tu veux)
    """

    def __init__(self,
                 R: float, r_max: float,
                 C_in: float, C_out: float,
                 D_in: float, D_out: float,
                 T1_in: float, T1_out: float,
                 P0_in: float, P0_out: float,
                 Tfinal: float = 40.0,
                 Nr: int = 500,
                 Nt: int = 500,
                 tanh_slope: float = 0.0,
                 solver: str = "lu_shift"  # "lu_shift" ou "gmres_ilu"
                 ):
        self.R = float(R); self.r_max = float(r_max)
        self.C_in, self.C_out = float(C_in), float(C_out)
        self.D_in, self.D_out = float(D_in), float(D_out)
        self.T1_in, self.T1_out = float(T1_in), float(T1_out)
        self.P0_in, self.P0_out = float(P0_in), float(P0_out)
        self.Tfinal = float(Tfinal); self.Nr = int(Nr); self.Nt = int(Nt)
        self.dt = self.Tfinal / max(1, self.Nt)
        self.tanh_slope = float(tanh_slope)
        self.solver = solver

        # Profils radiaux (marche nette ou tanh)
        if self.tanh_slope == 0.0:
            self.C  = lambda r: ufl.conditional(ufl.lt(r, self.R), self.C_in,  self.C_out)
            self.D  = lambda r: ufl.conditional(ufl.lt(r, self.R), self.D_in,  self.D_out)
            self.T1 = lambda r: ufl.conditional(ufl.lt(r, self.R), self.T1_in, self.T1_out)
            self.P0 = lambda r: ufl.conditional(ufl.lt(r, self.R), self.P0_in, self.P0_out)
        else:
            def _blend(a_in, a_out):
                mid = 0.5 * (a_out + a_in); amp = 0.5 * (a_out - a_in)
                return lambda r: mid + amp * ufl.tanh((r - self.R)/self.tanh_slope)
            self.C  = _blend(self.C_in,  self.C_out)
            self.D  = _blend(self.D_in,  self.D_out)
            self.T1 = _blend(self.T1_in, self.T1_out)
            self.P0 = _blend(self.P0_in, self.P0_out)

        # sorties (rank 0 uniquement)
        self.P_time = None; self.r_sorted = None; self.t_vec = None

    def solve(self):
        comm = MPI.COMM_WORLD
        msh = mesh.create_interval(comm, self.Nr, [0.0, self.r_max])
        V = fem.functionspace(msh, ("Lagrange", 1))

        # Radial coord. + poids régularisé & NORMALISÉ
        x = ufl.SpatialCoordinate(msh)[0]; r = x
        h = self.r_max / self.Nr
        eps_w = ScalarType((0.5*h)**2)
        w = (r**2 + eps_w) / ScalarType(self.r_max**2)

        # Planchers numériques
        C_expr  = ufl.max_value(self.C(r),  ScalarType(1.0))
        D_expr  = ufl.max_value(self.D(r),  ScalarType(1e-18))
        T1_expr = ufl.max_value(self.T1(r), ScalarType(1e-3))
        P0_func = self.P0(r)

        # Rampe P0
        t_now = fem.Constant(msh, ScalarType(0.0))
        T_ramp = max(self.Tfinal*0.02, 10*self.dt)
        P0_eff = ufl.conditional(t_now < T_ramp,
                                 P0_func * (t_now/ScalarType(T_ramp)),
                                 P0_func)

        # Inconnue / Test
        P_n1 = ufl.TrialFunction(V); v = ufl.TestFunction(V)
        P_n = fem.Function(V); P_n.x.array[:] = 0.0
        dx = ufl.dx; dt = ScalarType(self.dt)

        # Formes
        mass_lhs = (C_expr/dt)*w*P_n1*v*dx
        mass_rhs = (C_expr/dt)*w*P_n *v*dx
        diff_lhs = (D_expr*C_expr)*w*ufl.dot(ufl.grad(P_n1), ufl.grad(v))*dx
        reac_lhs = (C_expr/T1_expr)*w*P_n1*v*dx
        reac_rhs = (C_expr/T1_expr)*w*P0_eff*v*dx

        # Shift diagonal absolu (stabilité pivots)
        alpha = ScalarType(1e-10)
        a_form = mass_lhs + diff_lhs + reac_lhs + alpha*P_n1*v*dx
        L_form = mass_rhs + reac_rhs

        # Solveur
        if self.solver == "gmres_ilu":
            petsc_opts = {"ksp_type":"gmres","ksp_rtol":1e-10,"ksp_atol":1e-14,"pc_type":"ilu"}
        else:
            petsc_opts = {
                "ksp_type":"preonly","pc_type":"lu",
                "pc_factor_shift_type":"NONZERO",
                "pc_factor_shift_amount":1e-8
            }
        problem = LinearProblem(a_form, L_form, [], petsc_options=petsc_opts)

        # Collecte coord. DOF triées sur rank 0
        n_loc = V.dofmap.index_map.size_local
        r_loc = V.tabulate_dof_coordinates()[:n_loc, 0]
        r_all = msh.comm.gather(r_loc, root=0)
        if msh.comm.rank == 0:
            r_glob = np.concatenate(r_all)
            order = np.argsort(r_glob)
            self.r_sorted = r_glob[order]
            P_hist = [np.zeros_like(self.r_sorted)]

        # Boucle temps
        t = 0.0
        for _ in range(self.Nt):
            t += float(self.dt); t_now.value = ScalarType(t)
            P_new = problem.solve()

            arr_loc = P_new.x.array[:n_loc]
            if not np.isfinite(arr_loc).all():
                raise RuntimeError(f"Non-finite at t={t:.3e}s (local): "
                                   f"min={np.nanmin(arr_loc)}, max={np.nanmax(arr_loc)}")
            P_all = msh.comm.gather(arr_loc, root=0)
            if msh.comm.rank == 0:
                P_glob = np.concatenate(P_all)[order]
                if not np.isfinite(P_glob).all():
                    raise RuntimeError(f"Non-finite at t={t:.3e}s (global): "
                                       f"min={np.nanmin(P_glob)}, max={np.nanmax(P_glob)}")
                P_hist.append(P_glob.copy())

            P_n.x.array[:] = P_new.x.array

        if msh.comm.rank == 0:
            self.P_time = np.array(P_hist)
            self.t_vec  = np.linspace(0.0, self.Tfinal, self.Nt+1)


# ============================================================================
#                   OUTILS : MOYENNES SPHÉRIQUES & I/O
# ============================================================================
def spherical_averages(P_rt: np.ndarray, r: np.ndarray, R: float | None):
    """P̄(t) = ∫ P r^2 dr / ∫ r^2 dr  ; renvoie (Pavg_all, Pavg_in, Pavg_out)."""
    w = r**2
    denom_all = np.trapezoid(w, x=r)
    Pavg_all = np.trapezoid(P_rt*w, x=r, axis=1) / denom_all

    Pavg_in = Pavg_out = None
    if R is not None:
        idx_in  = r <= R
        idx_out = r >  R
        if idx_in.any():
            r_in = r[idx_in]; w_in = r_in**2
            denom_in = np.trapezoid(w_in, x=r_in)
            Pavg_in = np.trapezoid(P_rt[:, idx_in]*w_in, x=r_in, axis=1)/denom_in
        if idx_out.any():
            r_out = r[idx_out]; w_out = r_out**2
            denom_out = np.trapezoid(w_out, x=r_out)
            Pavg_out = np.trapezoid(P_rt[:, idx_out]*w_out, x=r_out, axis=1)/denom_out
    return Pavg_all, Pavg_in, Pavg_out


def save_npz(path: Path, P_rt, r, t, params: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, P_rt=P_rt, r=r, t=t, params=params)


# ============================================================================
#          LECTURE DU PICKLE (FORMAT {case_name: exp_data})
# ============================================================================
def _find_key(d: dict, candidates: list[str]) -> str:
    """Trouve la 1ère clé candidate (insensible à la casse) dans d, sinon lève."""
    lower = {k.lower(): k for k in d.keys()}
    for cand in candidates:
        k = lower.get(cand.lower())
        if k: return k
    raise KeyError(f"Clé introuvable. Attendues parmi {candidates}, disponibles: {list(d.keys())}")


def _get_first(d: dict, keys: list[str], *, required=True):
    """Renvoie la 1ère valeur présente parmi keys (insensible casse)."""
    lower = {k.lower(): k for k in d.keys()}
    for k in keys:
        real = lower.get(k.lower())
        if real in d: return d[real]
    if required:
        raise KeyError(f"Champs manquant, attendus {keys}, disponibles: {list(d.keys())}")
    return None


def compute_solver_params_from_exp(case_name: str, exp_data: dict) -> dict:
    """
    Reproduit **exactement** la logique de ton code :
      - solid_data_key = "CrisOn"
      - solvent_data_key = "JuiceOn"
      - D_f/j par loi (C/C_ref)^(1/3) avec D_ref=500 nm^2/s
      - R_s en nm → mètres
      - Tfinal = max(t) côté solide
      - tanh_slope = 0.05 * R
    Avec garde-fous et fallback de clés (si casse différente).
    """
    # Clés (tolérance sur la casse + quelques alias courants)
    solid_key   = _find_key(exp_data, ["CrisOn","SolidOn","Solid","Core","Nucleus"])
    solvent_key = _find_key(exp_data, ["JuiceOn","Solvent","Outside","Matrix"])

    # Paramètres "globaux" côté exp_data
    # R en mètres (R_s en nm dans ton code)
    if "R_s" in exp_data:
        R_m = float(exp_data["R_s"]) * 1e-9
    elif "R_nm" in exp_data:
        R_m = float(exp_data["R_nm"]) * 1e-9
    elif "R" in exp_data:
        R_m = float(exp_data["R"])  # suppose déjà en m
    else:
        raise KeyError(f"[{case_name}] rayon 'R_s' manquant dans exp_data.")

    # Concentrations & diffusion (mêmes constantes que ton code)
    C_ref = 60.0
    D_ref_nm2_s = 500.0
    D_ref_m2_s = D_ref_nm2_s * 1e-18

    C_f = float(exp_data.get("C_f", C_ref))
    C_j = float(exp_data.get("C_j", C_ref))
    D_f = D_ref_m2_s * ((C_f / C_ref)**(1.0/3.0))
    D_j = D_ref_m2_s * ((C_j / C_ref)**(1.0/3.0))

    # T1_in et T1_out
    T1_in  = float(exp_data.get("T_1_f", 300.0))
    T1_out = float(_get_first(exp_data[solid_key], ["TB_j","T1_j","T_out","TB"]))

    # P0_in = 1.0 (ton code), P0_out côté "JuiceOn"
    P0_in  = 1.0
    P0_out = float(_get_first(exp_data[solvent_key], ["P0_j","P0","P_out"]))

    # Temps final : max(t) côté solide
    t_solid = _get_first(exp_data[solid_key], ["t","time","times"])
    t_arr = np.asarray(t_solid, dtype=float).ravel()
    if t_arr.size == 0:
        raise ValueError(f"[{case_name}] pas de 't' côté {solid_key}.")
    Tfinal = float(np.max(t_arr))

    # Discrétisation (ton exemple : Nr=Nt=1000)
    Nr = int(exp_data.get("Nr", 1000))
    Nt = int(exp_data.get("Nt", 1000))

    params_solver = dict(
        R=R_m,
        r_max=R_m*6.0,
        C_in=C_f*1000.0, C_out=C_j*1000.0,
        D_in=D_f, D_out=D_j,
        T1_in=T1_in, T1_out=T1_out,
        P0_in=P0_in, P0_out=P0_out,
        Tfinal=Tfinal, Nr=Nr, Nt=Nt,
        tanh_slope=R_m*0,
        solver="lu_shift"
    )
    return params_solver


# ============================================================================
#                                   MAIN
# ============================================================================
def run_case(case_name: str, exp_data: dict, outdir: Path, comm: MPI.Comm):
    rank = comm.rank
    if rank == 0:
        print(f"\n--- Traitement du cas : {case_name} ---")
    t0 = time.time()

    # Paramètres FEM à partir du pickle 
    params_solver = compute_solver_params_from_exp(case_name, exp_data)

    # Solve
    gen = DataGenerator(**params_solver)
    gen.solve()  # garde-fous lèvent si non-fini

    row = None
    if gen.P_time is not None and rank == 0:
        # Chemin EXACT
        base = outdir / f"{case_name}_reference_FEM" / "Data"
        npz_path = base / "reference_FEM_solution.npz"
        save_npz(npz_path, gen.P_time, gen.r_sorted, gen.t_vec, params_solver)

        P = gen.P_time
        P_min, P_max = float(P.min()), float(P.max())
        Pavg_all, Pavg_in, Pavg_out = spherical_averages(P, gen.r_sorted, params_solver["R"])

        row = {
            "case": case_name,
            "P_min": P_min, "P_max": P_max,
            "Pavg_all_T": float(Pavg_all[-1]),
            "Pavg_in_T":  float(Pavg_in[-1])  if Pavg_in  is not None else np.nan,
            "Pavg_out_T": float(Pavg_out[-1]) if Pavg_out is not None else np.nan,
            "runtime_s": round(time.time()-t0, 3),
            # audit : paramètres principaux
            "R": params_solver["R"], "r_max": params_solver["r_max"],
            "C_in": params_solver["C_in"], "C_out": params_solver["C_out"],
            "D_in": params_solver["D_in"], "D_out": params_solver["D_out"],
            "T1_in": params_solver["T1_in"], "T1_out": params_solver["T1_out"],
            "P0_in": params_solver["P0_in"], "P0_out": params_solver["P0_out"],
            "Tfinal": params_solver["Tfinal"], "Nr": params_solver["Nr"], "Nt": params_solver["Nt"],
            "tanh_slope": params_solver["tanh_slope"],
        }
        print(f"   OK  → {npz_path}  (min={P_min:.4g}, max={P_max:.4g})")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True, help="Pickle {case_name: exp_data}")
    parser.add_argument("--output_dir", required=True, help="Dossier de sortie")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD; rank = comm.rank

    # I/O de base
    if rank == 0:
        outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
        print("="*56)
        print("Lancement FEM (multi-cas) – format donnees.pkl conforme à ton code")
        print(f"Fichier de données : {args.data_file}")
        print(f"Dossier de sortie  : {outdir}")
        print("="*56)
    else:
        outdir = None
    comm.Barrier()

    # Lecture pickle (dict {name: exp_data})
    if rank == 0:
        with open(args.data_file, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise TypeError("Le pickle doit être un dict {case_name: exp_data}.")
        print(f"Nombre de cas : {len(data)}")
    else:
        data = None
    data = comm.bcast(data, root=0)
    outdir = comm.bcast(outdir, root=0)

    # Lancer tous les cas
    rows = []
    for case_name, exp_data in data.items():
        try:
            row = run_case(case_name, exp_data, outdir, comm)
            if rank == 0 and row is not None:
                rows.append(row)
        except Exception as e:
            if rank == 0:
                print(f"   *** ERREUR {case_name}: {e}")
        comm.Barrier()

    # Récap CSV
    if rank == 0 and rows:
        csv_path = outdir / "recap_polarisation.csv"
        with open(csv_path, "w", newline="") as fcsv:
            fieldnames = list(rows[0].keys())
            w = csv.DictWriter(fcsv, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)
        print(f"\nRécap écrit : {csv_path}")
        print("Terminé")


if __name__ == "__main__":
    main()
