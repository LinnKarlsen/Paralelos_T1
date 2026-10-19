import pandas as pd
import sys

from funcs.run_c import run_c
from funcs.run_python import run_python

# Nombre del CSV a guardar
name = sys.argv[1] # vicho o linn
csv_name = f"resultados_{name}/resultados_escalabilidad.csv"

# Nombres de los ejecutables / scripts
binary1 = "algoritmo1"
binary2 = "algoritmo2"
binary3 = "algoritmo3.py"

N_list = [1024, 2048, 4096]

threads_list = [1, 2, 4, 8]

resultados = []

print("\nHOLA LINN REVISTATE EL n ÓPTIMO A USAR?????? antes de correr\n")

for N in N_list:

    # decide que n usar! yo creo que es el mismo
    n_data_parallelism = 512
    n_conquer = 128

    for p in threads_list:
        print(f"exec N={N}, p={p}...")

        # Correr Algoritmo 1 (C++ OpenMP - Datos)
        r_a1 = run_c(binary1, N, n_data_parallelism, p)
        
        # Correr Algoritmo 2 (C++ OpenMP - Dividir y Conquistar)
        r_a2 = run_c(binary2, N, n_conquer, p)
        
        # Correr Algoritmo 3 (Python Numba)
        r_a3 = run_python(binary3, N, n_data_parallelism, p)

        # Asegurar que el diccionario guarde N y p de forma explícita
        for r in [r_a1, r_a2, r_a3]:
            r["N"] = N
            r["threads"] = p

        # Guardar en la lista general
        resultados.append(r_a1)
        resultados.append(r_a2)
        resultados.append(r_a3)

# Convertir a DataFrame de Pandas y exportar a CSV
df = pd.DataFrame(resultados)
df.to_csv(csv_name, index=False)

# print bonito hola linn
print("csv guardado!!! hola linn")
