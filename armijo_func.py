import numpy as np

class Func:
    def get_fun(self, x):
        val = 0.0
        x_size = int(x.size / 2)
        for i in range(x_size):
            val = val + 100 * (x[2 * i] ** 2 - x[2 * i + 1]) ** 2 + (x[2 * i] - 1) ** 2
        return val[0]

    def get_grad(self, x):
        x_size = int(x.size / 2)
        grad = np.zeros((x.size, 1))
        for i in range(x_size):
            grad[2*i] = 400 * (x[2 * i] ** 2 - x[2 * i + 1]) * x[2 * i] + 2 * (x[2 * i] - 1)
            grad[2*i + 1] = -200 * (x[2 * i] ** 2 - x[2 * i + 1])
        return grad

    # def get_hession(self, x):
    #     x_size = int(x.size / 2)
    #     hession = np.zeros((x.size, x.size))
    #     for i in range(x_size):



