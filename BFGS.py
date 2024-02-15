import numpy as np

def get_BFGS_B(Bk, deltaG, deltaX):
    size = int(Bk.shape[0])
    I = np.eye(size)
    tmp = 1 / (deltaG.T @ deltaX)[0][0]
    B_next = (I - deltaX @ deltaG.T * tmp) * Bk * (I - deltaG @ deltaX.T * tmp) + (deltaX @ deltaX.T * tmp)
    return B_next

def get_Cautious_BFGS_B(Bk, deltaG, deltaX, g):
    size = int(Bk.shape[0])
    I = np.eye(size)
    tmp1 = (deltaG.T @ deltaX)[0][0]
    tmp2 = 1e-6 * np.linalg.norm(g, ord=2) * (deltaX.T @ deltaX)[0][0]
    B_next = Bk
    if (tmp1 > tmp2):
        tmp = 1 / tmp1
        B_next = (I - deltaX @ deltaG.T * tmp) @ Bk @ (I - deltaG @ deltaX.T * tmp) + (deltaX @ deltaX.T * tmp)
    return B_next