import pandas as pd

from funcs.run_c import run_c
from funcs.run_python import run_python

"""
Script adaptado para evaluar únicamente en los pares proporcionales (N, p) 
para el análisis de Escalabilidad Débil.
"""

# Nombre del CSV a guardar
csv_name = "resultados/resultados_escalabilidad_debil.csv"

# Nombres de los ejecutables / scripts
binary1 = "algoritmo1"
binary2 = "algoritmo2"
binary3 = "algoritmo3.py"

# Definir explícitamente los pares proporcionales (N, p) para escalabilidad débil
# Ejemplo: A medida que duplicamos los hilos p, duplicamos también el tamaño de la matriz N
pares_proporcionales = [
    {"p": 1, "N": 1024},
    {"p": 2, "N": 2048},
    {"p": 4, "N": 4096},
    # {"p": 8, "N": 8192} # Descomentar si tu hardware soporta este tamaño
]

resultados = []

print("\nHOLA LINN REVISTATE EL n ÓPTIMO A USAR?????? antes de correr\n")

# Iterar exclusivamente sobre la lista de pares proporcionales
for par in pares_proporcionales:
    p = par["p"]
    N = par["N"]
    
    # Definir el tamaño de bloque n proporcional o completo según corresponda
    n_data_parallelism = N
    n_conquer = N 

    print(f"exec N={N}, p={p}...")

    # Correr Algoritmo 1 (C++ OpenMP - Datos)
    r_a1 = run_c(binary1, N, n_data_parallelism, p)
    
    # Correr Algoritmo 2 (C++ OpenMP - Dividir y Conquistar)
    r_a2 = run_c(binary2, N, n_conquer, p)
    
    # Correr Algoritmo 3 (Python Numba)
    r_a3 = run_python(binary3, N, n_data_parallelism, p)

    # Asegurar que cada diccionario guarde N y p de forma explícita
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

print("csv guardado!!! hola linn")
