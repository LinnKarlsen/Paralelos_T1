import sys
import numpy as np
import matplotlib.pyplot as plt

from funciones.bs_auto import bs_auto
from funciones.bs_sklearn import bs_sklearn
from funciones.bs_numpy import bs_numpy
from funciones.funcs import gen_testbench

def plot_time(Tp, title):

    # eje x
    workers = np.arange(1, len(Tp) + 1)

    fig = plt.figure(figsize=(8, 5))

    # curva real
    plt.plot(workers, Tp, marker='o', linewidth=2, color='blue')

    # cositas
    plt.title(title)
    plt.xlabel("cantidad de workers")
    plt.ylabel("tiempo (s)")

    # mas cositas
    plt.xticks(workers)
    plt.grid(True)
    plt.tight_layout()

    return fig

def plot_speedup(Tp, title):

    # eje x
    workers = np.arange(1, len(Tp) + 1)

    # calcular speedup
    T1 = Tp[0]
    speedup = [T1 / tp for tp in Tp]

    fig = plt.figure(figsize=(8, 5))

    # curva real
    plt.plot(workers, speedup, marker='o', linewidth=2, color='blue', label='speedup real')

    # curva ideal (Sp = p)
    plt.plot(workers, workers, linestyle='--', color='green', label='speedup ideal')

    # cositas
    plt.title(title)
    plt.xlabel("cantidad de workers")
    plt.ylabel("speedup")

    # mas cositas
    plt.xticks(workers)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return fig

def plot_eficiencia(Tp, title):

    # eje x
    workers = np.arange(1, len(Tp) + 1)

    # calcular eficiencia
    T1 = Tp[0]
    eficiencia = [T1 / (p*tp) for tp in Tp]

    fig = plt.figure(figsize=(8, 5))

    # curva real
    plt.plot(workers, eficiencia, marker='o', linewidth=2, color='blue', label='eficiencia real')

    # curva ideal (Sp = p)
    plt.plot(workers, np.ones_like(workers), linestyle='--', color='green', label='eficiencia ideal')

    # cositas
    plt.title(title)
    plt.xlabel("cantidad de workers")
    plt.ylabel("eficiencia")

    # mas cositas
    plt.xticks(workers)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return fig

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
    tm_auto_fig = plot_time(auto_times, "tiempo bs_auto")
    tm_skelearn_fig = plot_time(sklearn_times, "tiempo bs_sklearn")
    tm_numpy_fig = plot_time(numpy_times, "tiempo bs_numpy")

    # guardar tiempos
    tm_auto_fig.savefig('graficos/tm_auto_fig.png', dpi=300)
    tm_skelearn_fig.savefig('graficos/tm_sklearn_fig.png', dpi=300)
    tm_numpy_fig.savefig('graficos/tm_numpy_fig.png', dpi=300)

    # speedups
    spdp_auto_fig = plot_speedup(auto_times, "speedup bs_auto")
    spdp_skelearn_fig = plot_speedup(sklearn_times, "speedup bs_sklearn")
    spdp_numpy_fig = plot_speedup(numpy_times, "speedup bs_numpy")

    # guardar speedups
    spdp_auto_fig.savefig('graficos/spdp_auto_fig.png', dpi=300)
    spdp_skelearn_fig.savefig('graficos/spdp_sklearn_fig.png', dpi=300)
    spdp_numpy_fig.savefig('graficos/spdp_numpy_fig.png', dpi=300)

    # eficiencia
    efcn_auto_fig = plot_eficiencia(auto_times, "eficiencia bs_auto")
    efcn_sklearn_fig = plot_eficiencia(sklearn_times, "eficiencia bs_sklearn")
    efcn_numpy_fig = plot_eficiencia(numpy_times, "eficiencia bs_numpy")

    # guardar eficiencias
    efcn_auto_fig.savefig('graficos/efcn_auto_fig.png', dpi=300)
    efcn_sklearn_fig.savefig('graficos/efcn_sklearn_fig.png', dpi=300)
    efcn_numpy_fig.savefig('graficos/efcn_numpy_fig.png', dpi=300)

    print("\nMALFE YA ESTA LISTO AVISALE AL VICHO\n")