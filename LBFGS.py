import numpy as np
import weak_wolfe_conditions

def get_L_BFGS_d(s, y, rho, g):
    d = g
    real_m = len(s)
    alpha = np.zeros(real_m)
    for i in range(real_m - 1, -1, -1):
        alpha[i] = rho[i] * (s[i].T @ d)[0][0]
        d = d - alpha[i] * y[i]
    gamma = rho[real_m - 1] * (y[real_m - 1].T @ y[real_m - 1])[0][0]
    d = d / gamma
    for i in range(0, real_m):
        beta = rho[i] * (y[i].T @ d)[0][0]
        d = d + s[i] * (alpha[i] - beta)

    return -d
