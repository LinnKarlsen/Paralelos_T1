import sys
import numpy as np
import matplotlib.pyplot as plt
from time import time
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from funciones.funcs import numpy_solve, gen_testbench, gen_idx

# numpy_solve con limitacion de threads
def numpy_solve_limited(X, y, idx, t_threads):
    with threadpool_limits(limits=t_threads, user_api='blas'):
        beta = numpy_solve(X, y, idx)
        return beta

# bs_numpy con limitacion de threads
def bs_numpy_limited(X, y, B, p, t):

    start = time()

    tasks = [delayed(numpy_solve_limited)(X, y, gen_idx(X.shape[0], i), t) for i in range(B)]
    with Parallel(n_jobs=p) as parallel_pool:
        _ = parallel_pool(tasks)
        
    end = time()

    return end - start

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

    tiempos_matrix = np.full((p_max, p_max), np.nan)

    for p in range(1, p_max + 1):
        for t in range(1, p_max + 1):
            if p * t <= p_max:
                t_exec = bs_numpy_limited(X, y, B, p, t)
                tiempos_matrix[p - 1, t - 1] = t_exec
                print(f"Combinación (p={p}, t={t}) -> Tiempo: {t_exec:.4f} s")

    fig, ax = plt.subplots(figsize=(8, 7))

    # color map
    # color gris a las celdas invalidas
    cmap = plt.cm.YlGnBu_r.copy()
    cmap.set_bad(color='#e0e0e0')

    # imshow
    im = ax.imshow(tiempos_matrix, cmap=cmap, origin='upper')

    # barra de color
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.set_ylabel("Tiempo (s)", rotation=-90, va="bottom", fontsize=11)

    # ejes discretos
    ax.set_xticks(np.arange(p_max))
    ax.set_yticks(np.arange(p_max))
    ax.set_xticklabels([f"{t}" for t in range(1, p_max + 1)])
    ax.set_yticklabels([f"{p}" for p in range(1, p_max + 1)])

    ax.set_xlabel("Cantidad de threads por worker (t)")
    ax.set_ylabel("Cantidad de workers (p)")
    ax.set_title(f"Tiempos para combinaciones p, t")

    plt.tight_layout()
    fig.savefig("graficos/parte_i.png", dpi=300)