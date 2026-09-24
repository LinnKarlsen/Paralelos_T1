import numpy as np
import sys

from funciones.bs_auto import bs_auto
from funciones.bs_sklearn import bs_sklearn
from funciones.bs_numpy import bs_numpy
from funciones.funcs import gen_testbench

if __name__ == "__main__":

    # parametros
    N = 10000
    k = 300
    B = 48
    p = int(sys.argv[1])
    verbose = int(sys.argv[2])

    # semilla
    MAIN_SEED = 42

    # generar X, y
    rng = np.random.default_rng(seed=MAIN_SEED)
    beta, X, y = gen_testbench(rng, k, N)

    ####### algoritmos de bootstrapping

    # bs_auto.py
    print("\nEjecutando función bs_auto...")
    coefs, bs_auto_time = bs_auto(X, y, B, p, MAIN_SEED, verbose = verbose)
    print(f"Tiempo total: {bs_auto_time}\n")

    # bs_auto.py
    print("Ejecutando función bs_sklearn...")
    coefs, bs_sklearn_time = bs_sklearn(X, y, B, p, verbose = verbose)
    print(f"Tiempo total: {bs_sklearn_time}\n")

    # bs_auto.py
    print("Ejecutando función bs_numpy...")
    coefs, bs_numpy_time = bs_numpy(X, y, B, p, verbose = verbose)
    print(f"Tiempo total: {bs_numpy_time}\n")