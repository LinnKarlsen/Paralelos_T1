from sklearn.linear_model import LinearRegression
import numpy as np
from joblib import Parallel, delayed
from time import perf_counter

# funcion que devuelve coeficientes de regresion
def fit_lr(X, y, seed):

    # qty de samples a generar
    n = X.shape[0]

    # hacer bootstrap
    # elegir n indices aleatorios entre 0 y n con reemplazo
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size = n, replace=True)

    # nuevos X, y con bootstrapped indices
    X_bs, y_bs = X[idx], y[idx]

    # crear y ajustar modelo
    base_lr = LinearRegression()
    base_lr.fit(X_bs, y_bs)

    coefs = base_lr.coef_

    return coefs

def bs_sklearn(X, y, B, p):

    start_global = perf_counter()

    tasks = [delayed(fit_lr)(X, y, seed) for seed in range(B)]
    with Parallel(n_jobs=p, verbose=1) as parallel_pool:
        parallel_results = parallel_pool(tasks)
    
    end_global = perf_counter()
    total_time = end_global - start_global

    return [parallel_results, total_time]