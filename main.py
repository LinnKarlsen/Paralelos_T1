import numpy as np
from joblib import Parallel, delayed
import threadpoolctl
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor

from funciones.bs_auto import bs_auto
from funciones.bs_sklearn import bs_sklearn

######################################################################################

# parametros
N = 100
k = 30
B = 48
p = 2

# semillas para que siempre salga igual
MAIN_SEED = 42
semillas = np.random.SeedSequence(MAIN_SEED).spawn(B)

#(I)

rnd = np.random.default_rng(seed=MAIN_SEED)
beta = rnd.normal(loc=0, scale=1, size=(k+1, 1))

#(II)

X = rnd.normal(loc=0, scale=1, size=(N, k+1))
#X=np.random.random(size=(N, k+1))
X[:,0]=1

#(III)

N = rnd.normal(loc=0, scale=1, size=(N, 1))
# N = np.random.random(size=(N, 1))
y = (X @ beta) + N

#### tiene que ser un vector normaaaaal
y = y.reshape(-1)


# El .shape de las matrices/vectores se expresa como (filas, columnas)
print(beta.shape)
print(X.shape)
print(y.shape)

######################################################################################

bs_auto(B, p, X, y, MAIN_SEED)

p = 2

bs_sklearn(X, y, B, p)

import multiprocessing

print(multiprocessing.cpu_count())
print("AMD Ryzen 7 de la serie 7000: 8 núcleos y 16 hilos")