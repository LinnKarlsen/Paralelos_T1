import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from time import time

def bs_auto(X, y, B, p, MAIN_SEED, verbose = 0):

    # modelo base para hacer input al bagging
    base_lr = LinearRegression()

    # modelo de baggingregressor
    bagging_regressor = BaggingRegressor(
        estimator=base_lr,
        n_estimators=B,
        bootstrap=True,
        n_jobs = p,
        verbose = verbose,
        random_state = MAIN_SEED
    )

    # medir
    start = time()

    # ajustar
    bagging_regressor.fit(X, y)

    # medir
    end = time()

    # extraer coeficientes
    estimadores = bagging_regressor.estimators_
    coefs = np.array([est.coef_ for est in estimadores])

    # convertimos devuelta a np.array
    coefs = np.vstack(coefs)

    return coefs, end-start