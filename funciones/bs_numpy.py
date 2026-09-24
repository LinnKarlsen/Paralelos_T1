import numpy as np
from joblib import Parallel, delayed
from time import time

from funciones.funcs import gen_idx

# funcion que devuelve coeficientes de regresion
def numpy_solve(X, y, idx):

    # nuevos X, y con bootstrapped indices
    X_bs, y_bs = X[idx], y[idx]

    # crear y ajustar modelo
    beta = np.linalg.solve(X_bs.T @ X_bs, X_bs.T @ y_bs)

    return beta

###
def bs_numpy(X, y, B, p, verbose = 0):

    # medir
    start = time()

    # calcular coefs
    tasks = [delayed(numpy_solve)(X, y, gen_idx(X.shape[0], i)) for i in range(B)]
    with Parallel(n_jobs=p, verbose=verbose) as parallel_pool:
        parallel_results = parallel_pool(tasks)

    # medir
    end = time()

    return parallel_results, end-start