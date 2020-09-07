import numpy as np
from numba import jit, guvectorize


@jit(nopython=True)
def seed(s):
    np.random.seed(s)


@guvectorize(["void(float64[:], int32[:])"], '(n)->()')
def choice(p, res):
    """按给定的概率从0~n-1中随机选择一个数并返回，p[i]表示i被选中的概率。"""
    x = np.random.rand()
    cum = 0
    for i, p in enumerate(p):
        cum += p
        if x < cum:
            res[0] = i
            break
