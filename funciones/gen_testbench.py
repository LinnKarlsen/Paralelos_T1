

def gen_testbench(rnd, k, N):
    #(I)

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

    return [beta, X, y]