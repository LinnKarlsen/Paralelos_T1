import numpy as np

# generar beta, X, y
def gen_testbench(rnd, k, N):

    # vector beta
    beta = rnd.normal(loc=0, scale=1, size=(k+1, 1))

    # matriz X
    X = rnd.normal(loc=0, scale=1, size=(N, k+1))
    X[:,0]=1

    # vector y + vector aleatorio
    N = rnd.normal(loc=0, scale=1, size=(N, 1))
    y = (X @ beta) + N
    y = y.reshape(-1)

    return beta, X, y

# generar bootstrapped indices
def gen_idx(n, SEED):

    # generador
    rng = np.random.default_rng(SEED)

    # n indices entre 0 y n-1 con repeticion
    idx = rng.choice(n, size = n, replace=True)
    
    return idx