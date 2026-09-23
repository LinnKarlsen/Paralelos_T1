from sklearn.linear_model import LinearRegression
import numpy as np
from joblib import Parallel, delayed

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

    tasks = [delayed(fit_lr)(X, y, seed) for seed in range(B)]
    with Parallel(n_jobs=p, verbose=1) as parallel_pool:
        parallel_results = parallel_pool(tasks)

    print(np.percentile(parallel_results, 2.5, axis = 0)) # lower bounds para vector de betas
    print(np.percentile(parallel_results, 97.5, axis = 0)) # upper bounds para vector de betas

    return