import numpy as np, glob, os

def check_npz(path):
    data = np.load(path, allow_pickle=True)
    P, r, t = data["P_rt"], data["r"], data["t"]
    print(f"\n→ {os.path.basename(path)}  P.shape={P.shape}, r={r.shape}, t={t.shape}")
    print("  finite? ", np.isfinite(P).all())
    print("  min/max:", np.nanmin(P), np.nanmax(P))
    return P, r, t, data["params"].item() if "params" in data else {}

for f in sorted(glob.glob("*.npz")):   # adapte le pattern si besoin
    check_npz(f)



def spherical_avg(P_rt, r, t, R=None):
    """Retourne:
       - Pavg_all(t) sur tout [0,rmax]
       - Pavg_in(t)  sur [0,R]  (si R fourni)
       - Pavg_out(t) sur [R,rmax] (si R fourni)
    """
    # Trapezes 1D avec poids r^2
    w = r**2
    def trapz_t(y):
        # y: (Nt+1, Nr) comme P_rt
        return np.trapz(y, x=r, axis=1)
    denom_all = np.trapz(w, x=r)
    Pavg_all = trapz_t(P_rt * w) / denom_all

    if R is None:
        return Pavg_all, None, None

    idx_in  = r <= R
    idx_out = r >  R
    # éviter les cas vides
    P_in  = P_rt[:, idx_in];   r_in  = r[idx_in];  w_in  = r_in**2
    P_out = P_rt[:, idx_out];  r_out = r[idx_out]; w_out = r_out**2

    denom_in  = np.trapz(w_in,  x=r_in)  if r_in.size  else np.nan
    denom_out = np.trapz(w_out, x=r_out) if r_out.size else np.nan

    Pavg_in  = (np.trapz(P_in  * w_in,  x=r_in,  axis=1)/denom_in)  if r_in.size  else None
    Pavg_out = (np.trapz(P_out * w_out, x=r_out, axis=1)/denom_out) if r_out.size else None
    return Pavg_all, Pavg_in, Pavg_out

# Exemple d’usage sur un fichier:
path = sorted(glob.glob("*.npz"))[0]
P, r, t, params = check_npz(path)
R = params.get("R", None)
Pavg_all, Pavg_in, Pavg_out = spherical_avg(P, r, t, R)

print("\nMoyennes au temps final:")
print("  P̄_all(T) =", float(Pavg_all[-1]))
if Pavg_in  is not None: print("  P̄_in(T)  =", float(Pavg_in[-1]))
if Pavg_out is not None: print("  P̄_out(T) =", float(Pavg_out[-1]))


import csv

rows = []
for f in sorted(glob.glob("*.npz")):
    P, r, t, params = check_npz(f)
    R = params.get("R", None)
    Pavg_all, Pavg_in, Pavg_out = spherical_avg(P, r, t, R)
    rows.append({
        "file": os.path.basename(f),
        "R": params.get("R"), "r_max": params.get("r_max"),
        "C_in": params.get("C_in"), "C_out": params.get("C_out"),
        "D_in": params.get("D_in"), "D_out": params.get("D_out"),
        "T1_in": params.get("T1_in"), "T1_out": params.get("T1_out"),
        "P0_in": params.get("P0_in"), "P0_out": params.get("P0_out"),
        "Tfinal": params.get("Tfinal"), "Nr": params.get("Nr"), "Nt": params.get("Nt"),
        "P_min": float(np.nanmin(P)), "P_max": float(np.nanmax(P)),
        "Pavg_all_T": float(Pavg_all[-1]),
        "Pavg_in_T":  float(Pavg_in[-1])  if Pavg_in  is not None else np.nan,
        "Pavg_out_T": float(Pavg_out[-1]) if Pavg_out is not None else np.nan,
    })

with open("recap_polarisation.csv", "w", newline="") as fcsv:
    w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\nCSV écrit : recap_polarisation.csv")
