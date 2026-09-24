import numpy as np

from funciones.bs_auto import bs_auto
from funciones.bs_sklearn import bs_sklearn
from funciones.bs_numpy import bs_numpy
from funciones.gen_funcs import gen_testbench

# Parametros
N = 100000
k = 300
B = 48
p = 8

# Semillas
MAIN_SEED = 42

# (a) Generar testbench utilizando la semilla principal

rnd = np.random.default_rng(seed=MAIN_SEED)
beta, X, y = gen_testbench(rnd, k, N)

# Revisamos tamaños
print("Tamaños en (filas, columnas):")
print("beta:", beta.shape)
print("X:   ", X.shape)
print("y:   ", y.shape)

# (b) Algoritmo de bootstrapping, implementado con diferentes metodologías

# Método 1: bs_auto.py
print(3*"\n")
print("Ejecutando función bs_auto...")

coefs = bs_auto(X, y, B, p, MAIN_SEED)

# eliminar 2.5% inferior y 2.5% superior para cada beta
#print(beta)
#print(np.percentile(coefs, 2.5, axis = 0))
#print(np.percentile(coefs, 97.5, axis = 0))


# METODO 2: bs_sklearn.py

print(3*"\n")
print("Ejecutando función bs_sklearn...")

p = 4 # Número de trabajadores

parallel_results, total_time = bs_sklearn(X, y, B, p)

print("Tiempo total transcurrido:",total_time)

#print(np.percentile(parallel_results, 2.5, axis = 0))  # cota inferior para vector de betas
#print(np.percentile(parallel_results, 97.5, axis = 0)) # cota superior para vector de betas
# MÉTODO 3: ...

# TODO