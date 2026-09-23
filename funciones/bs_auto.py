import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor

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

    # eliminar 2.5% inferior y 2.5% superior para cada beta
    print(np.percentile(coefs, 2.5, axis = 0))
    print(np.percentile(coefs, 97.5, axis = 0))