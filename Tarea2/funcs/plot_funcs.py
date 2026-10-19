import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 17})

def plot_time(Ta1, Ta2, Ta3):

    p_max = len(Ta1)

    # eje x
    workers = [1, 2, 4, 8]

    fig = plt.figure(figsize=(9, 7))

    # curvas
    plt.plot(workers, Ta1, marker='.', linewidth=2, label="paralelismo en loops")
    plt.plot(workers, Ta2, marker='.', linewidth=2, label="divide and conquer")
    plt.plot(workers, Ta3, marker='.', linewidth=2, label="numba en python")

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

def plot_speedup(Ta1, Ta2, Ta3):

    p_max = len(Ta1)

    # eje x
    workers = [1, 2, 4, 8]

    # calcular speedups
    r_a1 = [Ta1[0] / tp for tp in Ta1]
    r_a2 = [Ta2[0] / tp for tp in Ta2]
    r_a3 = [Ta3[0] / tp for tp in Ta3]

    fig = plt.figure(figsize=(9, 7))

    # curvas
    plt.plot(workers, r_a1, marker='.', linewidth=2, label="paralelismo en loops")
    plt.plot(workers, r_a2, marker='.', linewidth=2, label="divide and conquer")
    plt.plot(workers, r_a3, marker='.', linewidth=2, label="numba en python")

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

def plot_eficiencia(Ta1, Ta2, Ta3):
    
    p_max = len(Ta1)

    # eje x
    workers = [1, 2, 4, 8]

    # calcular eficiencias
    r_a1 = [Ta1[0] / (Ta1[i]*workers[i]) for i in range(p_max)]
    r_a2 = [Ta2[0] / (Ta2[i]*workers[i]) for i in range(p_max)]
    r_a3 = [Ta3[0] / (Ta3[i]*workers[i]) for i in range(p_max)]

    fig = plt.figure(figsize=(9, 7))

    # curvas
    plt.plot(workers, r_a1, marker='.', linewidth=2, label="paralelismo en loops")
    plt.plot(workers, r_a2, marker='.', linewidth=2, label="divide and conquer")
    plt.plot(workers, r_a3, marker='.', linewidth=2, label="numba en python")

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

def plot_overhead(Ta1, Ta2, Ta3, T1):
    
    p_max = len(Ta1)

    # eje x
    workers = [1, 2, 4, 8]

    # calcular overheads
    r_a1 = [(i+1)*Ta1[i]-T1 for i in range(p_max)]
    r_a2 = [(i+1)*Ta2[i]-T1 for i in range(p_max)]
    r_a3 = [(i+1)*Ta3[i]-T1 for i in range(p_max)]

    fig = plt.figure(figsize=(9, 7))

    # curvas
    plt.plot(workers, r_a1, marker='.', linewidth=2, label="paralelismo en loops")
    plt.plot(workers, r_a2, marker='.', linewidth=2, label="divide and conquer")
    plt.plot(workers, r_a3, marker='.', linewidth=2, label="numba en python")

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

def plot_vs_n(T):

    n_vec = [2048, 1024, 512, 256, 128, 64, 32]

    fig = plt.figure(figsize=(9, 7))

    # curvas
    plt.plot(n_vec, T, marker='.', linewidth=2)

    # cositas
    plt.title("Tiempo de ejecución según tamaño de bloque")
    plt.xlabel("Tamaño de bloque")
    plt.ylabel("Tiempo (s)")

    # mas cositas
    plt.xticks(n_vec)
    plt.grid(True)
    plt.tight_layout()

    return fig

def plot_eficiencia_debil_simple(T_weak, workers):
    """
    Plotea la eficiencia débil para un solo algoritmo a partir de una lista de tiempos
    medidos en tamaños proporcionales (N, p).
    
    Parámetros:
    - T_weak: lista con los tiempos de ejecución ordenados por worker [T(p=1), T(p=2), T(p=4), ...]
    - workers: lista con la cantidad de workers evaluada [1, 2, 4, 8]
    """
    p_max = len(T_weak)

    # Cálculo de la Eficiencia Débil: E_débil = T_1(N_base) / T_p(N_escalado)
    # T_weak[0] es el tiempo base con p=1 y el tamaño inicial N
    eficiencia_debil = [T_weak[0] / T_weak[i] for i in range(p_max)]

    fig = plt.figure(figsize=(8, 5))

    # Curva de eficiencia débil experimental
    plt.plot(workers, eficiencia_debil, marker='.', linewidth=2)

    # Línea de referencia ideal (Eficiencia Débil Ideal = 1.0 constante)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Eficiencia Débil Ideal (1.0)')

    # Configuración del gráfico
    plt.title("Eficiencia débil para pares (p, N) proporcionales")
    plt.xlabel("Cantidad de workers")
    plt.ylabel("Eficiencia débil")

    # Ajustes de ejes y grilla
    plt.xticks(workers)
    plt.ylim(0, 1.25)
    plt.legend(fontsize=11)
    plt.tight_layout()

    return fig
