import pandas as pd

from funcs.run_c import run_c
from funcs.run_python import run_python

"""
formato del csv (para que todos los algoritmos queden igual):
binary,N,p

el nombre "binary" es el nombre del EJECUTABLE que corresponde al
codigo COMPILADO

NO CORRER main.py HASTA QUE ESTEN LOS ALGORITMOS LISTOS Y COMPILADOS
CON SUS NOMBRES BIEN PUESTOS
"""

# nombre del csv a guardar
csv_name = "resultados.csv"

# nombres de los ejecutables
binary1 = "algoritmo1"
binary2 = "algoritmo2"
binary3 = "algoritmo3.py"

# tamaño
N = 64

# cantidad de threads
threads_list = [1, 2, 4, 8]

### ciclo sobre cantidad de threads
resultados = [] # para guardar resultados
for p in threads_list:

    # correr codigos de c
    r_a1 = run_c(binary1, N, p) # en c
    r_a2 = run_c(binary2, N, p) # en c
    r_a3 = run_python(binary3, N, p) # en python? hay que hacer otra funcion

    # guardar
    resultados.append(r_a1)
    resultados.append(r_a2)
    resultados.append(r_a3)

# data frame + csv
df = pd.DataFrame(resultados)
df.to_csv(csv_name, index=False)

# print bonito hola linn
print("csv guardado!!! hola linn")
