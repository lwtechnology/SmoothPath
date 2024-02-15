import numpy as np
import math

class Func:
    def get_fun(self, x):
        val =  (1 - x[0]) ** 2 + math.fabs(x[1] - x[0] ** 2)
        return val[0]

    def get_grad(self, x):
        grad = np.zeros((2, 1))
        if x[1] >= x[0] ** 2:
            grad[0] = -2 * (1 - x[0]) -2 * x[0]
            grad[1] = 1.0
        else :
            grad[0] = -2 * (1 - x[0]) + 2 * x[0]
            grad[1] = -1.0
        return grad



