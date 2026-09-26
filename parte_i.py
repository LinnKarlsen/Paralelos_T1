import sys
import numpy as np
from time import time
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from funciones.funcs import numpy_solve, fit_lr, gen_idx, gen_testbench

# numpy_solve con limitacion de threads
def numpy_solve_limited(X, y, idx, t_threads):
    with threadpool_limits(limits=t_threads, user_api='blas'):
        beta = numpy_solve(X, y, idx)
        return beta

def bs_numpy_limited(X, y, B, p, t):
    """
    Ejecuta el algoritmo bootstrap usando p procesos (workers)
    y t hilos nativos por worker.
    """
    start = time()
    
    # Generador de tareas para Parallel
    tasks = [
        delayed(numpy_solve_worker)(X, y, np.random.choice(X.shape[0], X.shape[0]), t) 
        for _ in range(B)
    ]
    
    # Ejecución paralela con p procesos (backend 'loky' por defecto)
    with Parallel(n_jobs=p) as parallel_pool:
        results = parallel_pool(tasks)
        
    end = time()
    return end - start

# =====================================================================
# 2. Configuración de la grilla de experimentos (p, t) con p * t <= p_max
# =====================================================================
p_max = 8  # Modificar según la cantidad de núcleos físicos de tu sistema (ej: M1 -> 8)
B_samples = 200  # Cantidad de muestras de bootstrap para los experimentos

# Matrices/Tablas para registrar resultados (filas = p, columnas = t)
tiempos_matrix = np.full((p_max, p_max), np.nan)

print("Iniciando matriz de experimentos (p, t)...")
for p in range(1, p_max + 1):
    for t in range(1, p_max + 1):
        # Condición restrictiva exigida: p * t <= p_max
        if p * t <= p_max:
            # Medición de tiempo
            t_exec = bs_numpy_experimento(X, y, B=B_samples, p=p, t=t)
            tiempos_matrix[p - 1, t - 1] = t_exec
            print(f"Combinación (p={p}, t={t}) -> Tiempo: {t_exec:.4f} s")

# =====================================================================
# 3. Visualización de Resultados mediante Mapa de Calor (Heatmap)
# =====================================================================
plt.figure(figsize=(9, 7))

### HEATMAP

plt.title(f"Tiempos de Ejecución para Combinaciones (p, t) con $p \cdot t \le {p_max}$", fontsize=13, pad=12)
plt.xlabel("Número de Threads Internos por Worker ($t$)", fontsize=11)
plt.ylabel("Número de Procesos / Workers ($p$)", fontsize=11)
plt.grid(False)
plt.tight_layout()
plt.show()

# =====================================================================
# 4. Identificación de la Mejor Combinación
# =====================================================================
# Obtener las coordenadas del tiempo mínimo omitiendo valores NaN
p_opt_idx, t_opt_idx = np.unravel_index(np.nanargmin(tiempos_matrix), tiempos_matrix.shape)
p_opt, t_opt = p_opt_idx + 1, t_opt_idx + 1
tiempo_opt = tiempos_matrix[p_opt_idx, t_opt_idx]

print("\n" + "="*50)
print(f"RESULTADO OPTIMO:")
print(f"Mejor configuración: p = {p_opt} procesos, t = {t_opt} hilos/process")
print(f"Tiempo mínimo alcanzado: {tiempo_opt:.4f} segundos")
print("="*50)


if __name__ == "__main__":

    # parametros
    N = 1000
    k = 30
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