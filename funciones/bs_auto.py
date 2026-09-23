import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from time import perf_counter

def bs_auto(B, p, X, y, MAIN_SEED):

    base_lr = LinearRegression()

    # modelo de baggingregressor
    bagging_regressor = BaggingRegressor(
        estimator=base_lr,
        n_estimators=B,
        bootstrap=True,
        n_jobs = p,
        verbose = 1,
        random_state = MAIN_SEED
    )

    # ajustar
    bagging_regressor.fit(X, y)

    # extraer coeficientes
    estimadores = bagging_regressor.estimators_
    coefs = np.array([est.coef_ for est in estimadores])

    return coefs