import numpy as np

def get_weak_wolfe_step(x, d, testFunc):
    c1 = 1e-4
    c2 = 0.9
    curr_grad = testFunc.get_grad(x)
    tau = 1.0
    x_new = x + tau * d
    tmp1 = c2 * (d.T @ curr_grad)[0][0]
    tmp2 = (d.T @ testFunc.get_grad(x_new))[0][0]
    while (testFunc.get_fun(x_new) > testFunc.get_fun(x) + c1 * tau * (d.T @ curr_grad)[0][0] or
    tmp1 > tmp2):
        tau = 0.5 * tau
        x_new = x + tau * d
        tmp1 = c2 * (d.T @ curr_grad)[0][0]
        tmp2 = (d.T @ testFunc.get_grad(x_new))[0][0]
    return tau