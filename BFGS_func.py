import numpy as np
import math

class Func:
    def get_fun(self, x):
        val = (1 - x[0]) ** 2 + (x[1] - x[0] ** 2) ** 2
        return val[0]

    def get_grad(self, x):
        grad = np.zeros((2, 1))
        grad[0] = -2 * (1 - x[0]) + 2 * (x[1] - x[0] ** 2) * (-2 * x[0])
        grad[1] = 2 * (x[1] - x[0] ** 2)
        return grad

    def get_hession(self, x):
        hession = np.zeros((2, 2))
        hession[0][0] = 2 - 4 * x[1] + 12 * x[0]**2
        hession[1][0] = -4 * x[0]
        hession[0][1] = hession[1][0]
        hession[1][1] = 2
        return hession




