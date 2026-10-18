import pandas as pd
from funcs.run_c import run_c

"""
formato del csv (para que todos los algoritmos queden igual):
binary,N,p

el nombre "binary" es el nombre del EJECUTABLE que corresponde al
codigo COMPILADO

NO CORRER main.py HASTA QUE ESTEN LOS ALGORITMOS LISTOS Y COMPILADOS
CON SUS NOMBRES BIEN PUESTOS
"""

# nombre del csv a guardar
csv_name = "resultados/conquer_time_vs_n.csv"

# nombres de los ejecutables
binary = "algoritmo2"

# tamaño
N = 2048
p = 1

# cantidad de threads
n_list = [2048, 1024, 512, 256, 128, 64, 32]

### ciclo sobre cantidad de threads
resultados = [] # para guardar resultados
for n in n_list:

    r = run_c(binary, N, n, p) # en python? hay que hacer otra funcion

    # guardar
    resultados.append(r)

# data frame + csv
df = pd.DataFrame(resultados)
df.to_csv(csv_name, index=False)

# print bonito hola vicho
print("csv guardado!!! hola vicho")