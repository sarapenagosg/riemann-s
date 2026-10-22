import numpy as np
import pandas as pd
from riemann_ml.geometry import sample_branch_points, branch_gaps

def family_interior(n, seed, min_gap=0.08):
    rng =  np.random.default_rng(seed)
    filas = []
    for i in range(n):
        e = sample_branch_points(rng, min_gap)
        g = branch_gaps(e)

        fila = {
            "sample_id": f"INT{i+1:05d}",
            "family_id": "interior",
            "seed": seed,
        }
        for j, ej in enumerate(e, start=1):
            fila[f"e{j}"] = ej
        for j, gj in enumerate(g, start=1):
            fila[f"g{j}"] = gj
        fila["t1"] = np.nan
        fila["t2"] = np.nan
        fila["provisional_label"] = False

        filas.append(fila)
    return pd.DataFrame(filas)

def family_nonseparating(t_values, a=0.42, b=0.72):
    filas = []
    for i, t in enumerate(t_values):
        if not (0 < t < a < b < 1):
            raise ValueError(f"t={t} no cumple 0 < t < a < b < 1")

        e = np.array([0,t,a,b,1,2])
        g = branch_gaps(e)

        fila = {
            "sample_id": f"NSP{i+1:05d}",
            "family_id": "nonseparating",
            "seed": np.nan,
        }
        for j, ej in enumerate(e, start=1):
            fila[f"e{j}"] = ej
        for j, gj in enumerate(g, start=1):
            fila[f"g{j}"] = gj
        fila["t1"] = t
        fila["t2"] = np.nan
        fila["provisional_label"] = False

        filas.append(fila)
    return pd.DataFrame(filas)

def family_separating(eps_values, config):
    raise NotImplementedError(
        "Familia separante provisional: pendiente de validación matemática del generador."
    )