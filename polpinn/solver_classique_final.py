#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_solver.py
Lit un fichier pickle avec une liste de cas, résout chaque cas en FEM (dolfinx/PETSc),
calcule la polarisation moyenne (globale / cœur / extérieur), et enregistre :
 - <output_dir>/<case_name>.npz  (P_rt, r, t, params)
 - <output_dir>/recap_polarisation.csv

GPU : géré via PETSc si l'environnement exporte:
  -vec_type cuda/hip -mat_type aijcusparse/aijhipsparse -dm_mat_type aijcusparse/aijhipsparse
(voir le script SLURM que tu as déjà)

Dépendances: mpi4py, numpy, dolfinx, petsc4py, ufl
"""
from __future__ import annotations
import os, sys, csv, argparse, pickle, time
import numpy as np
from mpi4py import MPI

import ufl
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem


# =========================================================
#                CLASSE FEM ROBUSTE
# =========================================================
class DataGenerator:
    """
    Équation : C(r) dP/dt = (1/r^2) d/dr [ r^2 D(r) C(r) dP/dr ] - C(r) (P - P0(r)) / T1(r)
    Discrétisation 1D FEM (P1) avec stabilisations :
      - poids régularisé & NORMALISÉ : w = (r^2 + eps) / r_max^2
      - rampe sur P0
      - planchers numériques (C,D,T1)
      - diagonal shift (stabilité LU) + pivot shift (configurable via PETSc)
    """

    def __init__(self,
                 R: float, r_max: float,
                 C_in: float, C_out: float,
                 D_in: float, D_out: float,
                 T1_in: float, T1_out: float,
                 P0_in: float, P0_out: float,
                 Tfinal: float = 10.0,
                 Nr: int = 100,
                 Nt: int = 100,
                 tanh_slope: float = 0.0,
                 solver: str = "lu_shift"  # "lu_shift" (défaut) ou "gmres_ilu"
                 ):
        self.R = float(R)
        self.r_max = float(r_max)
        self.C_in, self.C_out = float(C_in), float(C_out)
        self.D_in, self.D_out = float(D_in), float(D_out)
        self.T1_in, self.T1_out = float(T1_in), float(T1_out)
        self.P0_in, self.P0_out = float(P0_in), float(P0_out)
        self.Tfinal = float(Tfinal)
        self.Nr = int(Nr)
        self.Nt = int(Nt)
        self.dt = self.Tfinal / max(1, self.Nt)
        self.tanh_slope = float(tanh_slope)
        self.solver = solver

        if self.tanh_slope == 0.0:
            self.C  = lambda r: ufl.conditional(ufl.lt(r, self.R), self.C_in,  self.C_out)
            self.D  = lambda r: ufl.conditional(ufl.lt(r, self.R), self.D_in,  self.D_out)
            self.T1 = lambda r: ufl.conditional(ufl.lt(r, self.R), self.T1_in, self.T1_out)
            self.P0 = lambda r: ufl.conditional(ufl.lt(r, self.R), self.P0_in, self.P0_out)
        else:
            def _tanh_blend(a_in, a_out):
                mid = 0.5 * (a_out + a_in)
                amp = 0.5 * (a_out - a_in)
                return lambda r: mid + amp * ufl.tanh((r - self.R) / self.tanh_slope)
            self.C  = _tanh_blend(self.C_in,  self.C_out)
            self.D  = _tanh_blend(self.D_in,  self.D_out)
            self.T1 = _tanh_blend(self.T1_in, self.T1_out)
            self.P0 = _tanh_blend(self.P0_in, self.P0_out)

        self.P_time = None
        self.r_sorted = None
        self.t_vec = None

    def solve(self):
        comm = MPI.COMM_WORLD
        msh = mesh.create_interval(comm, self.Nr, [0.0, self.r_max])
        V = fem.functionspace(msh, ("Lagrange", 1))

        x = ufl.SpatialCoordinate(msh)[0]
        r = x
        h = self.r_max / self.Nr
        eps_w = ScalarType((0.5 * h) ** 2)
        w = (r**2 + eps_w) / ScalarType(self.r_max**2)  # normalisé

        C_expr  = ufl.max_value(self.C(r),  ScalarType(1.0))
        D_expr  = ufl.max_value(self.D(r),  ScalarType(1e-18))
        T1_expr = ufl.max_value(self.T1(r), ScalarType(1e-3))
        P0_func = self.P0(r)

        t_now = fem.Constant(msh, ScalarType(0.0))
        T_ramp = max(self.Tfinal * 0.02, 10 * self.dt)
        P0_eff = ufl.conditional(t_now < T_ramp,
                                 P0_func * (t_now / ScalarType(T_ramp)),
                                 P0_func)

        P_n1 = ufl.TrialFunction(V)
        v    = ufl.TestFunction(V)
        P_n  = fem.Function(V)
        P_n.x.array[:] = 0.0

        dx = ufl.dx
        dt = ScalarType(self.dt)

        mass_lhs = (C_expr / dt) * w * P_n1 * v * dx
        mass_rhs = (C_expr / dt) * w * P_n  * v * dx
        diff_lhs = (D_expr * C_expr) * w * ufl.dot(ufl.grad(P_n1), ufl.grad(v)) * dx
        reac_lhs = (C_expr / T1_expr) * w * P_n1 * v * dx
        reac_rhs = (C_expr / T1_expr) * w * P0_eff * v * dx

        alpha = ScalarType(1e-10)  # diagonal shift absolu
        a_form = mass_lhs + diff_lhs + reac_lhs + alpha * P_n1 * v * dx
        L_form = mass_rhs + reac_rhs

        if self.solver == "gmres_ilu":
            petsc_opts = {
                "ksp_type": "gmres",
                "ksp_rtol": 1e-10,
                "ksp_atol": 1e-14,
                "pc_type": "ilu",
            }
        else:
            petsc_opts = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_shift_type": "NONZERO",
                "pc_factor_shift_amount": 1e-8,
            }

        problem = LinearProblem(a_form, L_form, [], petsc_options=petsc_opts)

        n_loc = V.dofmap.index_map.size_local
        r_loc = V.tabulate_dof_coordinates()[:n_loc, 0]
        r_all = msh.comm.gather(r_loc, root=0)
        if msh.comm.rank == 0:
            r_glob = np.concatenate(r_all)
            order = np.argsort(r_glob)
            self.r_sorted = r_glob[order]
            P_hist = [np.zeros_like(self.r_sorted)]

        t = 0.0
        for _ in range(self.Nt):
            t += float(self.dt)
            t_now.value = ScalarType(t)

            P_new = problem.solve()

            arr_loc = P_new.x.array[:n_loc]
            if not np.isfinite(arr_loc).all():
                raise RuntimeError(
                    f"Non-finite detected at t={t:.3e}s on local rank: "
                    f"min={np.nanmin(arr_loc)}, max={np.nanmax(arr_loc)}"
                )

            P_all = msh.comm.gather(arr_loc, root=0)
            if msh.comm.rank == 0:
                P_glob = np.concatenate(P_all)[order]
                if not np.isfinite(P_glob).all():
                    raise RuntimeError(
                        f"Non-finite detected at t={t:.3e}s (global): "
                        f"min={np.nanmin(P_glob)}, max={np.nanmax(P_glob)}"
                    )
                P_hist.append(P_glob.copy())

            P_n.x.array[:] = P_new.x.array

        if msh.comm.rank == 0:
            self.P_time = np.array(P_hist)
            self.t_vec = np.linspace(0.0, self.Tfinal, self.Nt + 1)


# =========================================================
#          POST-TRAITEMENT & STOCKAGE
# =========================================================
def spherical_averages(P_rt: np.ndarray, r: np.ndarray, t: np.ndarray, R: float | None):
    """Moyenne volumique sphérique: P̄(t) = ∫ P r^2 dr / ∫ r^2 dr."""
    w = r**2

    def trap_t(Y, x):
        # remplace trapz (deprecated)
        return np.trapezoid(Y, x=x, axis=1)

    denom_all = np.trapezoid(w, x=r)
    Pavg_all = trap_t(P_rt * w, r) / denom_all

    Pavg_in = Pavg_out = None
    if R is not None:
        idx_in = r <= R
        idx_out = r > R
        if idx_in.any():
            r_in = r[idx_in]; w_in = r_in**2
            denom_in = np.trapezoid(w_in, x=r_in)
            Pavg_in = np.trapezoid(P_rt[:, idx_in] * w_in, x=r_in, axis=1) / denom_in
        if idx_out.any():
            r_out = r[idx_out]; w_out = r_out**2
            denom_out = np.trapezoid(w_out, x=r_out)
            Pavg_out = np.trapezoid(P_rt[:, idx_out] * w_out, x=r_out, axis=1) / denom_out

    return Pavg_all, Pavg_in, Pavg_out


def save_npz(path: str, P_rt: np.ndarray, r: np.ndarray, t: np.ndarray, params: dict):
    np.savez(path, P_rt=P_rt, r=r, t=t, params=params)


# =========================================================
#             LECTURE DU PICKLE & NORMALISATION
# =========================================================
def load_cases_from_pickle(pkl_path: str) -> list[dict]:
    """Accepte plusieurs formats possibles :
       - liste de dicts
       - dict {"cases": [...]} 
       - dict {name: params_dict, ...}
       - dict d'un seul cas (on l'emballe en liste)
    Chaque dict doit contenir au minimum: R, r_max, C_in, C_out, D_in, D_out,
    T1_in, T1_out, P0_in, P0_out. Les champs optionnels: Tfinal, Nr, Nt, tanh_slope, solver, name.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    def is_case_dict(d):
        return isinstance(d, dict) and any(k in d for k in ("R", "R_nm", "params"))

    cases = []
    if isinstance(data, list):
        if all(isinstance(x, dict) for x in data):
            cases = data
        else:
            raise ValueError("Liste non reconnue dans le pickle (attendu: liste de dicts).")
    elif isinstance(data, dict):
        if "cases" in data and isinstance(data["cases"], list):
            cases = data["cases"]
        elif all(isinstance(v, dict) for v in data.values()):
            # mapping {name: params}
            for name, params in data.items():
                params = params.copy()
                params.setdefault("name", str(name))
                cases.append(params)
        elif is_case_dict(data):
            cases = [data]
        else:
            raise ValueError("Structure du pickle non reconnue.")
    else:
        raise ValueError("Type de pickle non supporté.")

    # Normalisation des champs + défauts
    normed = []
    for i, c in enumerate(cases):
        c = c.copy()

        # certains pickles mettent tout sous "params"
        if "params" in c and isinstance(c["params"], dict):
            base = c.get("params", {}).copy()
            for k in c:
                if k != "params":
                    base.setdefault(k, c[k])
            c = base

        # gestion unités alternatives (nm => m), si présentes
        if "R_nm" in c and "R" not in c:
            c["R"] = float(c["R_nm"]) * 1e-9
        if "r_max_nm" in c and "r_max" not in c:
            c["r_max"] = float(c["r_max_nm"]) * 1e-9

        required = ["R","r_max","C_in","C_out","D_in","D_out","T1_in","T1_out","P0_in","P0_out"]
        missing = [k for k in required if k not in c]
        if missing:
            raise KeyError(f"Cas #{i}: champs manquants {missing}")

        # défauts raisonnables
        c.setdefault("Tfinal", 40.0)
        c.setdefault("Nr", 500)
        c.setdefault("Nt", 500)
        c.setdefault("tanh_slope", 0.0)  # ou 2.5e-9 si tu veux une transition douce
        c.setdefault("solver", "lu_shift")

        # nom du cas
        if "name" not in c:
            c["name"] = (
                f"R{c['R']:.2e}_rmax{c['r_max']:.2e}_Cin{c['C_in']:.0f}_Cout{c['C_out']:.0f}"
                f"_T1in{c['T1_in']:.0f}_T1out{c['T1_out']:.0f}_P0in{c['P0_in']:.0f}_P0out{c['P0_out']:.0f}"
                f"_T{int(c['Tfinal'])}_Nr{c['Nr']}_Nt{c['Nt']}"
            ).replace("+","").replace("-","").replace(".","p")
        normed.append(c)

    return normed


# =========================================================
#                       MAIN
# =========================================================
def run_case(case: dict, outdir: str, comm: MPI.Comm):
    rank = comm.rank
    if rank == 0:
        print(f"\n--- Cas : {case['name']} ---")
    t0 = time.time()

    gen = DataGenerator(
        R=case["R"], r_max=case["r_max"],
        C_in=case["C_in"], C_out=case["C_out"],
        D_in=case["D_in"], D_out=case["D_out"],
        T1_in=case["T1_in"], T1_out=case["T1_out"],
        P0_in=case["P0_in"], P0_out=case["P0_out"],
        Tfinal=case["Tfinal"], Nr=case["Nr"], Nt=case["Nt"],
        tanh_slope=case["tanh_slope"],
        solver=case.get("solver", "lu_shift")
    )

    gen.solve()  # exceptions levées si non-fini

    row = None
    if gen.P_time is not None:  # rank 0
        npz_path = os.path.join(outdir, f"{case['name']}.npz")
        params = {k: case[k] for k in case}
        save_npz(npz_path, gen.P_time, gen.r_sorted, gen.t_vec, params)

        P = gen.P_time
        P_min, P_max = float(P.min()), float(P.max())

        Pavg_all, Pavg_in, Pavg_out = spherical_averages(P, gen.r_sorted, gen.t_vec, case["R"])
        row = {
            "case": case["name"],
            "P_min": P_min, "P_max": P_max,
            "Pavg_all_T": float(Pavg_all[-1]),
            "Pavg_in_T":  float(Pavg_in[-1])  if Pavg_in  is not None else np.nan,
            "Pavg_out_T": float(Pavg_out[-1]) if Pavg_out is not None else np.nan,
            "runtime_s": round(time.time() - t0, 3),
            # recopions aussi les params pour audit
            **{k: v for k, v in case.items() if k != "name"},
        }
        print(f"   OK  → {npz_path}  (min={P_min:.4g}, max={P_max:.4g})")

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True, help="Pickle avec les cas (donnees.pkl)")
    parser.add_argument("--output_dir", required=True, help="Dossier de sortie")
    parser.add_argument("--max_cases", type=int, default=None, help="Limiter le nb de cas (debug)")
    args = parser.parse_args()


    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Chargement des cas depuis: {args.data_file}")
    comm.Barrier()

    try:
        cases = load_cases_from_pickle(args.data_file)
    except Exception as e:
        if rank == 0:
            print(f"*** ERREUR lecture pickle: {e}")
        sys.exit(1)

    if args.max_cases is not None:
        cases = cases[:args.max_cases]

    if rank == 0:
        print(f"Nombre de cas à traiter: {len(cases)}")

    rows = []
    for case in cases:
        try:
            row = run_case(case, args.output_dir, comm)
            if rank == 0 and row is not None:
                rows.append(row)
        except Exception as e:
            if rank == 0:
                print(f"   *** ERREUR cas {case.get('name','<sans nom>')}: {e}")
        comm.Barrier()

    if rank == 0 and rows:
        csv_path = os.path.join(args.output_dir, "recap_polarisation.csv")
        with open(csv_path, "w", newline="") as fcsv:
            fieldnames = list(rows[0].keys())
            w = csv.DictWriter(fcsv, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nRécap écrit : {csv_path}")
        print("Terminé ✅")


if __name__ == "__main__":
    main()
