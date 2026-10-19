import pandas as pd
import sys
from funcs.run_serial import run_serial

"""
formato del csv (para que todos los algoritmos queden igual):
binary,N,p

el nombre "binary" es el nombre del EJECUTABLE que corresponde al
codigo COMPILADO

NO CORRER main.py HASTA QUE ESTEN LOS ALGORITMOS LISTOS Y COMPILADOS
CON SUS NOMBRES BIEN PUESTOS
"""

# nombre del csv a guardar
name = sys.argv[1]
csv_name = f"resultados_{name}/serial_time_vs_n.csv"

# nombres de los ejecutables
binary = "serial"

# tamaño
N = 2048

# cantidad de threads
n_list = [2048, 1024, 512, 256, 128, 64, 32]

### ciclo sobre cantidad de threads
resultados = [] # para guardar resultados
for n in n_list:

    r = run_serial(binary, N, n) # en python? hay que hacer otra funcion

    # guardar
    resultados.append(r)

# data frame + csv
df = pd.DataFrame(resultados)
df.to_csv(csv_name, index=False)

# print bonito hola vicho
print("csv guardado!!! hola vicho")