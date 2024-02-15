import numpy as np
import math

class Func:
    def get_fun(self, x):
        return math.exp(x[0] + 3 * x[1] - 0.1) + math.exp(x[0] - 3 * x[1] - 0.1) + math.exp(-x[0] - 0.1)

    def get_grad(self, x):
        grad = np.zeros((2, 1))
        grad[0] = math.exp(-0.1) * (math.exp(x[0] + 3 * x[1]) + math.exp(x[0] - 3 * x[1]) - math.exp(-x[0]))
        grad[1] = math.exp(-0.1) * (3 * math.exp(x[0] + 3 * x[1]) - 3 * math.exp(x[0] - 3 * x[1]))
        return grad

    def get_hession(self, x):
        hession = np.zeros((2, 2))
        hession[0][0] = math.exp(-0.1) * (math.exp(x[0] + 3 * x[1]) + math.exp(x[0] - 3 * x[1]) + math.exp(-x[0]))
        hession[1][0] = math.exp(-0.1) * (3 * math.exp(x[0] + 3 * x[1]) - 3 * math.exp(x[0] - 3 * x[1]))
        hession[0][1] = hession[1][0]
        hession[1][1] = math.exp(-0.1) * (9 * math.exp(x[0] + 3 * x[1]) + 9 * math.exp(x[0] - 3 * x[1]))
        return hession




