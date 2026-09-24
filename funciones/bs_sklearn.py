from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from time import time
import numpy as np

from funciones.funcs import gen_idx

# funcion que devuelve coeficientes de regresion
def fit_lr(X, y, idx):

    # nuevos X, y con bootstrapped indices
    X_bs, y_bs = X[idx], y[idx]

    # crear y ajustar modelo
    base_lr = LinearRegression()
    base_lr.fit(X_bs, y_bs)

    # sacar coeficientes
    coefs = base_lr.coef_

    return coefs

### 
def bs_sklearn(X, y, B, p, verbose = 0):

    # medir
    start = time()

    # calcular coefs
    tasks = [delayed(fit_lr)(X, y, gen_idx(X.shape[0], i)) for i in range(B)]
    with Parallel(n_jobs=p, verbose=verbose) as parallel_pool:
        parallel_results = parallel_pool(tasks)

    # medir
    end = time()

    # convertimos devuelta a np.array
    parallel_results = np.vstack(parallel_results)

    return parallel_results, end-start