from joblib import Parallel, delayed
from time import time
import numpy as np

from funciones.funcs import gen_idx, fit_lr

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