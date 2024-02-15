import numpy as np

def get_armijo_step(x, d, testFunc):
    c = 0.4
    curr_grad = testFunc.get_grad(x)
    tau = 1.0
    while testFunc.get_fun(x + tau * d) >  testFunc.get_fun(x) + c * tau * np.dot(d.T, curr_grad)[0][0]:
        tau = 0.5 * tau
    return tau