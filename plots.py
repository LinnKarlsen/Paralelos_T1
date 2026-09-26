import sys
import numpy as np
import matplotlib.pyplot as plt

from funciones.bs_auto import bs_auto
from funciones.bs_sklearn import bs_sklearn
from funciones.bs_numpy import bs_numpy
from funciones.funcs import gen_testbench, non_parallel

def plot_time(Tauto, Tsklearn, Tnumpy):

    p_max = len(Tauto)

    # eje x
    workers = np.arange(1, p_max + 1)

    fig = plt.figure(figsize=(8, 5))

    # curvas
    plt.plot(workers, Tauto, marker='.', linewidth=2, label="bs_auto")
    plt.plot(workers, Tsklearn, marker='.', linewidth=2, label="bs_sklearn")
    plt.plot(workers, Tnumpy, marker='.', linewidth=2, label="bs_numpy")

    # cositas
    plt.title("Tiempos de ejecución (T(p))")
    plt.xlabel("Cantidad de workers")
    plt.ylabel("Tiempo (s)")

    # mas cositas
    plt.xticks(workers)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return fig

def plot_speedup(Tauto, Tsklearn, Tnumpy):

    p_max = len(Tauto)

    # eje x
    workers = np.arange(1, p_max + 1)

    # calcular speedups
    sp_auto = [Tauto[0] / tp for tp in Tauto]
    sp_sklearn = [Tsklearn[0] / tp for tp in Tsklearn]
    sp_numpy = [Tnumpy[0] / tp for tp in Tnumpy]

    fig = plt.figure(figsize=(8, 5))

    # curvas
    plt.plot(workers, sp_auto, marker='.', linewidth=2, label="bs_auto")
    plt.plot(workers, sp_sklearn, marker='.', linewidth=2, label="bs_sklearn")
    plt.plot(workers, sp_numpy, marker='.', linewidth=2, label="bs_numpy")

    # speedup ideal
    plt.plot(workers, workers, linestyle='--', label='eficiencia ideal')

    # cositas
    plt.title("Speedup (S(p))")
    plt.xlabel("Cantidad de workers")
    plt.ylabel("Speedup")

    # mas cositas
    plt.xticks(workers)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return fig

def plot_eficiencia(Tauto, Tsklearn, Tnumpy):
    
    p_max = len(Tauto)

    # eje x
    workers = np.arange(1, p_max + 1)

    # calcular eficiencias
    ef_auto = [Tauto[0] / ((i+1)*Tauto[i]) for i in range(p_max)]
    ef_sklearn = [Tsklearn[0] / ((i+1)*Tsklearn[i]) for i in range(p_max)]
    ef_numpy = [Tnumpy[0] / ((i+1)*Tnumpy[i]) for i in range(p_max)]

    fig = plt.figure(figsize=(8, 5))

    # curvas
    plt.plot(workers, ef_auto, marker='.', linewidth=2, label="bs_auto")
    plt.plot(workers, ef_sklearn, marker='.', linewidth=2, label="bs_sklearn")
    plt.plot(workers, ef_numpy, marker='.', linewidth=2, label="bs_numpy")

    # eficiencia ideal
    plt.plot(workers, np.ones_like(workers), linestyle='--', label='eficiencia ideal')

    # cositas
    plt.title("Eficiencia (E(p))")
    plt.xlabel("Cantidad de workers")
    plt.ylabel("Eficiencia")

    # mas cositas
    plt.xticks(workers)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return fig

def plot_overhead(Tauto, Tsklearn, Tnumpy, T1):
    
    p_max = len(Tauto)

    # eje x
    workers = np.arange(1, p_max + 1)

    # calcular overheads
    oh_auto = [(i+1)*Tauto[i]-T1 for i in range(p_max)]
    oh_sklearn = [(i+1)*Tsklearn[i]-T1 for i in range(p_max)]
    oh_numpy = [(i+1)*Tnumpy[i]-T1 for i in range(p_max)]

    fig = plt.figure(figsize=(8, 5))

    # curvas
    plt.plot(workers, oh_auto, marker='.', linewidth=2, label="bs_auto")
    plt.plot(workers, oh_sklearn, marker='.', linewidth=2, label="bs_sklearn")
    plt.plot(workers, oh_numpy, marker='.', linewidth=2, label="bs_numpy")

    # overhead ideal
    plt.plot(workers, np.zeros_like(workers), linestyle='--', label='overhead ideal')

    # cositas
    plt.title("Overhead (T_o(p))")
    plt.xlabel("Cantidad de workers")
    plt.ylabel("Overhead (s)")

    # mas cositas
    plt.xticks(workers)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return fig

if __name__ == "__main__":

    # parametros
    N = 10000
    k = 300
    B = 48
    p_max = int(sys.argv[1])
    verbose = int(sys.argv[2])

    # semilla
    MAIN_SEED = 42

    # generar X, y
    rng = np.random.default_rng(seed=MAIN_SEED)
    beta, X, y = gen_testbench(rng, k, N)

    # tiempo sin paralelizar
    non_parallel_time = non_parallel(X, y, B)

    # listas de tiempos para plotear despues
    auto_times = []
    sklearn_times = []
    numpy_times = []

    # iterar sobre cantidad de workers
    for p in range(1, p_max + 1):

        # medir tiempos
        _, bs_auto_time = bs_auto(X, y, B, p, MAIN_SEED, verbose = verbose)
        _, bs_sklearn_time = bs_sklearn(X, y, B, p, verbose = verbose)
        _, bs_numpy_time = bs_numpy(X, y, B, p, verbose = verbose)

        # agregarlos a la lista de tiempos
        auto_times.append(bs_auto_time)
        sklearn_times.append(bs_sklearn_time)
        numpy_times.append(bs_numpy_time)

    # tiempos
    tm_fig = plot_time(auto_times, sklearn_times, numpy_times)
    tm_fig.savefig('graficos/tm_fig.png', dpi=300)

    # speedups
    sp_fig = plot_speedup(auto_times, sklearn_times, numpy_times)
    sp_fig.savefig('graficos/sp_fig.png', dpi=300)

    # eficiencias
    ef_fig = plot_eficiencia(auto_times, sklearn_times, numpy_times)
    ef_fig.savefig('graficos/ef_fig.png', dpi=300)

    # tiempos
    oh_fig = plot_overhead(auto_times, sklearn_times, numpy_times, non_parallel_time)
    oh_fig.savefig('graficos/oh_fig.png', dpi=300)