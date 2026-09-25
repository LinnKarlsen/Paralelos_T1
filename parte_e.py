import sys
import numpy as np
from time import time
from joblib import Parallel, delayed
from threadpoolctl import threadpool_info

from funciones.bs_numpy import numpy_solve
from funciones.funcs import gen_testbench, gen_idx

""" 
ESTO SE CORRE CON:
python3.13 parte_e.py p_max verbose

ejemplo (8 nucleos maximo sin hablar):
python3.13 parte_e.py 8 0
"""


# override de las funcion original para que quede con info
def numpy_solve_threadpool(X, y, idx):

    # beta
    beta = numpy_solve(X, y, idx)

    # info de threadpool
    info = threadpool_info()
    print(f"num_threads (de threadpool): {info[0]['num_threads']}")
    
    return beta

# override para que use la nueva numpy_solve
# agregue parametro backend para hacer expermientos
def bs_numpy_threadpool(X, y, B, p, verbose = 0, backend = "loky"):

    # medir
    start = time()

    # calcular coefs
    tasks = [delayed(numpy_solve_threadpool)(X, y, gen_idx(X.shape[0], i)) for i in range(B)]
    with Parallel(n_jobs=p, verbose=verbose, backend=backend) as parallel_pool:
        parallel_results = parallel_pool(tasks)

    # medir
    end = time()

    # convertimos devuelta a np.array
    parallel_results = np.vstack(parallel_results)

    return parallel_results, end-start


if __name__ == "__main__":

    # parametros
    N = 100000
    k = 300
    B = 48
    p_max = int(sys.argv[1])
    verbose = int(sys.argv[2])

    # semilla
    MAIN_SEED = 42

    # generar X, y
    rng = np.random.default_rng(seed=MAIN_SEED)
    beta, X, y = gen_testbench(rng, k, N)

    # BACKEND
    # si hay oversubscription, revisar diferencia entre:
    backend = "loky"
    #backend = "multiprocessing"

    # iterar sobre cantidad de workers
    for p in range(1, p_max + 1):
        print(f"\ncantidad de workers: {p}")

        # usar la version con info
        coefs_numpy, bs_numpy_time = bs_numpy_threadpool(X, y, B, p, verbose = verbose)

        # tiempo de ejecucion
        print(f"tiempo total: {bs_numpy_time}\n")