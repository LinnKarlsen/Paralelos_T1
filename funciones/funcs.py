import numpy as np
from time import time
from sklearn.linear_model import LinearRegression

# funcion que devuelve coeficientes de regresion con LinearRegression
def fit_lr(X, y, idx):

    # nuevos X, y con bootstrapped indices
    X_bs, y_bs = X[idx], y[idx]

    # crear y ajustar modelo
    base_lr = LinearRegression()
    base_lr.fit(X_bs, y_bs)

    # sacar coeficientes
    coefs = base_lr.coef_

    return coefs

# funcion que devuelve coeficientes de regresion con numpy
def numpy_solve(X, y, idx):

    # nuevos X, y con bootstrapped indices
    X_bs, y_bs = X[idx], y[idx]

    # crear y ajustar modelo
    beta = np.linalg.solve(X_bs.T @ X_bs, X_bs.T @ y_bs)

    return beta

# generar beta, X, y
def gen_testbench(rng, k, N):

    # vector beta
    beta = rng.normal(loc=0, scale=1, size=(k+1, 1))

    # matriz X
    X = rng.normal(loc=0, scale=1, size=(N, k+1))
    X[:,0] = 1

    # vector y + vector aleatorio
    randvec = rng.normal(loc=0, scale=1, size=(N, 1))
    y = (X @ beta) + randvec
    y = y.reshape(-1)

    return beta, X, y

# generar bootstrapped indices
def gen_idx(n, SEED):

    # generador
    rng = np.random.default_rng(SEED)

    # n indices entre 0 y n-1 con repeticion
    idx = rng.choice(n, size = n, replace=True)
    
    return idx

# retorna intervalos de confianza
def conf_interval(coefs, coef_idx):

    cota_inf = np.percentile(coefs[:, coef_idx], 2.5, axis = 0)  # cota inferior
    cota_sup = np.percentile(coefs[:, coef_idx], 97.5, axis = 0) # cota superior

    return cota_inf, cota_sup

def compare_conf_interval(coefs_auto, coefs_sklearn, coefs_numpy, beta, coef_idx):

    auto_inf, auto_sup = conf_interval(coefs_auto, coef_idx)
    sklearn_inf, sklearn_sup = conf_interval(coefs_sklearn, coef_idx)
    numpy_inf, numpy_sup = conf_interval(coefs_numpy, coef_idx)

    print(f"Intervalos de confianza para el coeficiente {coef_idx}, con valor exacto {beta[coef_idx][0]}:")
    print(f"bs_auto:    ({auto_inf}, {auto_sup})")
    print(f"bs_sklearn: ({sklearn_inf}, {sklearn_sup})")
    print(f"bs_numpy:   ({numpy_inf}, {numpy_sup})")
    print()

# calcular todo sin paralelizar
def non_parallel(X, y, B):

    # medir
    start = time()

    # algoritmo sin paralelizar
    for i in range(B):
        idx = gen_idx(X.shape[0], i)
        _ = numpy_solve(X, y, idx)

    # medir
    end = time()

    return end-start