import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

SEED=20260905
rng = np.random.default_rng(SEED)    #este seria solo el generador de numeros
min_gap = 0.08

def sample_branch_points(rng, min_gap):
    while True:
        lambdas = np.sort(rng.uniform(0,1,3))    #limites, tamaño 
        e= np.concatenate([[0],lambdas,[1],[2]])

        gaps= np.diff(e)

        if np.any(gaps < min_gap):
            continue
        else:
            break
    return e

def polynomial_coefficients(e):
    y2=Polynomial.fromroots(e)
    return y2.coef



filas = []
for i in range(500):
    ramas = sample_branch_points(rng, min_gap)
    coefs = polynomial_coefficients(ramas)

    fila = {
        "sample_id": f"S{i+1:05d}",
        "family_id": "interior",
        "seed": SEED,
    }
    for j, e in enumerate(ramas, start=1):
        fila[f"e{j}"] = e
    for j, c in enumerate(coefs):
        fila[f"c{j}"] = c
    fila["valid"] = True

    filas.append(fila)


df = pd.DataFrame(filas)
df