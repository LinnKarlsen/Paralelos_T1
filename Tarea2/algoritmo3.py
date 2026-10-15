import math
import time
import sys
import numpy as np
from numba import jit, njit, prange
import time

def det_value(i, j, offset):
    return math.sin(i * 0.1 + j * 0.3 + offset)


def init_matrix(N, offset):
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return np.sin(i * 0.1 + j * 0.3 + offset)

@njit(parallel=True)
def multiply_leaf(A, B, C):
    """
    C += A @ B
    """
    s = C.shape[0]

    for i in prange(s):
        for k in range(s):
            a = A[i, k]

            for j in range(s):
                C[i, j] += a * B[k, j]


@njit
def multiply_rec(A, B, C, n):
    """
    Recursive blocked multiplication:
    C += A * B
    """

    s = C.shape[0]

    if s <= n:
        multiply_leaf(A, B, C)
        return

    h = s // 2

    A00 = A[:h, :h]
    A01 = A[:h, h:]
    A10 = A[h:, :h]
    A11 = A[h:, h:]

    B00 = B[:h, :h]
    B01 = B[:h, h:]
    B10 = B[h:, :h]
    B11 = B[h:, h:]

    C00 = C[:h, :h]
    C01 = C[:h, h:]
    C10 = C[h:, :h]
    C11 = C[h:, h:]

    multiply_rec(A00, B00, C00, n)
    multiply_rec(A01, B10, C00, n)

    multiply_rec(A00, B01, C01, n)
    multiply_rec(A01, B11, C01, n)

    multiply_rec(A10, B00, C10, n)
    multiply_rec(A11, B10, C10, n)

    multiply_rec(A10, B01, C11, n)
    multiply_rec(A11, B11, C11, n)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 256

    A = init_matrix(N, 1.0)
    B = init_matrix(N, 2.0)
    C = np.zeros((N, N), dtype=np.float64)

    t0 = time.perf_counter()
    multiply_rec(A, B, C, n)
    t1 = time.perf_counter()

    checksum = np.sum(C)

    print(
        f"N={N} n={n} "
        f"tiempo={(t1-t0)*1000:.2f} ms "
        f"checksum={checksum:.6f}"
    )


if __name__ == "__main__":
    main()
    main()