import numpy as np
import matplotlib.pyplot as plt

from funciones.bs_auto import bs_auto
from funciones.bs_sklearn import bs_sklearn
from funciones.gen_testbench import gen_testbench

# Parametros
N = 100
k = 30
B = 48
p = 2

# Semillas
MAIN_SEED = 42
semillas = np.random.SeedSequence(MAIN_SEED).spawn(B)

# (a) Generar testbench utilizando la semilla principal

rnd = np.random.default_rng(seed=MAIN_SEED)
[beta, X, y] = gen_testbench(rnd, k, N)

# Revisamos tamaños
print("Tamaños en (filas, columnas):")
print("beta:", beta.shape)
print("X:   ", X.shape)
print("y:   ", y.shape)

# (b) Algoritmo de bootstrapping, implementado con diferentes metodologías

# Método 1: bs_auto.py

print("Ejecutando función bs_auto...")

coefs = bs_auto(B, p, X, y, MAIN_SEED)

# eliminar 2.5% inferior y 2.5% superior para cada beta
print(np.percentile(coefs, 2.5, axis = 0))
print(np.percentile(coefs, 97.5, axis = 0))

# METODO 2: bs_sklearn.py

print("Ejecutando función bs_sklearn...")

p = 4 # Número de trabajadores

[parallel_results, total_time] = bs_sklearn(X, y, B, p)

print("Tiempo total transcurrido:",total_time)

print(np.percentile(parallel_results, 2.5, axis = 0))  # cota inferior para vector de betas
print(np.percentile(parallel_results, 97.5, axis = 0)) # cota superior para vector de betas

# MÉTODO 3: ...

# TODO