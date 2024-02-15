import numpy as np

def get_lewis_overton_line_search_step(x, d, testFunc):
    c1 = 1e-4
    c2 = 0.9
    curr_grad = testFunc.get_grad(x)
    alpha = 1.0
    x_new = x + alpha * d
    val = testFunc.get_fun(x)
    s_alpha_right = -c1 * (d.T @ curr_grad)[0][0]
    c_alpha_right = c2 * (d.T @ curr_grad)[0][0]
    val_new = testFunc.get_fun(x_new)
    g_new = testFunc.get_grad(x_new)

    l = 0.0
    u = float('inf')
    while True:
        if val - val_new < alpha * s_alpha_right :
            u = alpha
        elif (d.T @ g_new)[0][0] < c_alpha_right:
            l = alpha
        else:
            return alpha

        if u != float('inf'):
            alpha = (l + u) * 0.5
        else:
            alpha = 2 * l

        x_new = x + alpha * d
        val_new = testFunc.get_fun(x_new)
        g_new = testFunc.get_grad(x_new)

    return alpha