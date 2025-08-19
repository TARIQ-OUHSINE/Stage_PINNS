# data_generator.py
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem


class DataGenerator:
    """
    Génère P(r,t) en 1D radial pour :
        C(r) dP/dt = (1/r^2) d/dr [ r^2 D(r) C(r) dP/dr ] - C(r) (P - P0(r)) / T1(r)

    Discrétisation :
      - FEM P1 sur [0, r_max]
      - Forme faible multipliée par w = r^2 + eps_w (eps_w ~ h^2 / 4)
      - Euler implicite en temps
      - Stabilisations numériques : planchers, rampe P0, diagonal shift
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
                 tanh_slope: float = 0.0):
        # paramètres
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

        # Profils radiaux (saut net ou transition tanh)
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

        # sorties (rank 0)
        self.P_time = None   # (Nt+1, Ndof triés selon r)
        self.r_sorted = None
        self.t_vec = None

    def solve(self):
        # --- Maillage et espace ---
        msh = mesh.create_interval(MPI.COMM_WORLD, self.Nr, [0.0, self.r_max])
        V = fem.functionspace(msh, ("Lagrange", 1))

        # --- Coordonnée radiale & poids régularisé ---
        x = ufl.SpatialCoordinate(msh)[0]
        r = x
        h = self.r_max / self.Nr
        eps_w = ScalarType((0.5 * h) ** 2)    # ~ h^2 / 4 pour éviter la ligne centrale singulière
        w = r**2 + eps_w

        # --- Coeffs + planchers numériques ---
        C_raw, D_raw, T1_raw = self.C(r), self.D(r), self.T1(r)
        C_expr  = ufl.max_value(C_raw,  ScalarType(1.0))     # C >= 1
        D_expr  = ufl.max_value(D_raw,  ScalarType(1e-18))   # D >= 1e-18
        T1_expr = ufl.max_value(T1_raw, ScalarType(1e-3))    # T1 >= 1e-3
        P0_func = self.P0(r)

        # --- Rampe sur P0 (anti-choc initial) ---
        t_now = fem.Constant(msh, ScalarType(0.0))
        T_ramp = max(self.Tfinal * 0.02, 10 * self.dt)
        P0_eff = ufl.conditional(t_now < T_ramp,
                                 P0_func * (t_now / ScalarType(T_ramp)),
                                 P0_func)

        # --- Inconnues/tests ---
        P_n1 = ufl.TrialFunction(V)
        v    = ufl.TestFunction(V)
        P_n  = fem.Function(V)
        P_n.x.array[:] = 0.0

        dx = ufl.dx
        dt = ScalarType(self.dt)

        # --- Formes (Euler implicite) ---
        mass_lhs = (C_expr / dt) * w * P_n1 * v * dx
        mass_rhs = (C_expr / dt) * w * P_n  * v * dx
        diff_lhs = (D_expr * C_expr) * w * ufl.dot(ufl.grad(P_n1), ufl.grad(v)) * dx
        reac_lhs = (C_expr / T1_expr) * w * P_n1 * v * dx
        reac_rhs = (C_expr / T1_expr) * w * P0_eff * v * dx

        # Diagonal shift (pour éviter pivots nuls / améliorer le conditionnement)
        alpha = ScalarType(1e-12)
        a_form = mass_lhs + diff_lhs + reac_lhs + alpha * w * P_n1 * v * dx
        L_form = mass_rhs + reac_rhs

        # --- Problème linéaire robuste ---
        # Option 1 (par défaut) : GMRES + ILU
        petsc_opts = {
            "ksp_type": "gmres",
            "ksp_rtol": 1e-10,
            "ksp_atol": 1e-14,
            "pc_type": "ilu",
        }
        # Option 2 (si tu préfères LU) : décommente ci-dessous
        # petsc_opts = {
        #     "ksp_type": "preonly",
        #     "pc_type": "lu",
        #     "pc_factor_shift_type": "NONZERO",
        #     "pc_factor_shift_amount": 1e-12,
        # }

        problem = LinearProblem(a_form, L_form, [], petsc_options=petsc_opts)

        # --- Prépare collecte des dofs triés sur rank 0 ---
        n_loc = V.dofmap.index_map.size_local
        r_loc = V.tabulate_dof_coordinates()[:n_loc, 0]
        r_all = msh.comm.gather(r_loc, root=0)
        if msh.comm.rank == 0:
            r_glob = np.concatenate(r_all)
            order = np.argsort(r_glob)
            self.r_sorted = r_glob[order]
            P_hist = [np.zeros_like(self.r_sorted)]

        # --- Boucle en temps ---
        t = 0.0
        for _ in range(self.Nt):
            t += float(self.dt)
            t_now.value = ScalarType(t)

            P_new = problem.solve()

            # Garde-fou : on coupe net si non-fini
            arr_loc = P_new.x.array[:n_loc]
            if not np.isfinite(arr_loc).all():
                mmin = np.nanmin(arr_loc)
                mmax = np.nanmax(arr_loc)
                raise RuntimeError(f"Non-finite detected at t={t:.3e}s on local rank: "
                                   f"min={mmin}, max={mmax}")

            # rassembler sur rank 0
            P_all = msh.comm.gather(arr_loc, root=0)
            if msh.comm.rank == 0:
                P_glob = np.concatenate(P_all)[order]
                if not np.isfinite(P_glob).all():
                    mmin = np.nanmin(P_glob)
                    mmax = np.nanmax(P_glob)
                    raise RuntimeError(f"Non-finite detected at t={t:.3e}s (global): "
                                       f"min={mmin}, max={mmax}")
                P_hist.append(P_glob.copy())

            # avance en temps
            P_n.x.array[:] = P_new.x.array

        # --- Sorties (rank 0) ---
        if msh.comm.rank == 0:
            self.P_time = np.array(P_hist)  # (Nt+1, Ndof triés)
            self.t_vec = np.linspace(0.0, self.Tfinal, self.Nt + 1)




from importlib import reload
import data_generator
reload(data_generator)
from data_generator import DataGenerator

print("solve present? ", hasattr(DataGenerator, "solve"))  # doit être True

gen = DataGenerator(
    R=5e-8, r_max=3e-7,
    C_in=5.8e4, C_out=1.1e4,
    D_in=4.943815468908251e-16, D_out=2.840428201431225e-16,
    T1_in=300.0, T1_out=3.0,
    P0_in=1.0, P0_out=100.0,
    Tfinal=40.0, Nr=500, Nt=500,
    tanh_slope=2.5e-9,
)
gen.solve()

# Rank 0 : sauver les résultats
if gen.P_time is not None:
    np.savez("solution_ref.npz", P_rt=gen.P_time, r=gen.r_sorted, t=gen.t_vec, params=dict(
        R=gen.R, r_max=gen.r_max, C_in=gen.C_in, C_out=gen.C_out,
        D_in=gen.D_in, D_out=gen.D_out, T1_in=gen.T1_in, T1_out=gen.T1_out,
        P0_in=gen.P0_in, P0_out=gen.P0_out, Tfinal=gen.Tfinal, Nr=gen.Nr, Nt=gen.Nt,
        tanh_slope=gen.tanh_slope,
    ))
