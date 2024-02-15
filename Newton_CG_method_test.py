import BFGS_func
import LBFGS
import Armijo
import weak_wolfe_conditions
from matplotlib import pyplot as plt
import numpy as np
import time

testFunc = BFGS_func.Func()

x = np.zeros((2, 1))
x[0] = 2.2
x[1] = -1.3

print("initial x: ", x)

x_tmp = []

maxIter = 100000
iter = 0

start_time = time.time()

while iter < maxIter:
    currGrad = testFunc.get_grad(x)
    norm_2 = np.linalg.norm(currGrad, ord=2)
    x_tmp.append([x[0][0], x[1][0], testFunc.get_fun(x)])
    if norm_2 < 1e-15:
        print("iter: ",iter, "get min val: ", testFunc.get_fun(x))
        print("opt x: ", x)
        break
    g = currGrad
    epi = min(1, norm_2) / 10
    d = np.zeros((2, 1))
    v = -g
    u = v
    cg_iter = 1
    while np.linalg.norm(v, ord=2) > epi * norm_2:
        if u.T @ (-g) <= 0:
            if cg_iter == 1:
                d = -g
            break

        alpha = (np.linalg.norm(v, ord=2)) ** 2 / (u.T @ (-g))
        d = d + alpha * u
        v_next = v - alpha * (-g)
        beta = (np.linalg.norm(v_next, ord=2)) ** 2 / (np.linalg.norm(v, ord=2)) ** 2
        u = v_next + beta * u
        v = v_next
        cg_iter += 1
    iter += 1

    t = Armijo.get_armijo_step(x, d, testFunc)
    x = x + t * d

# 记录结束时间
end_time = time.time()

# 计算执行时间（秒）
execution_time_seconds = end_time - start_time

# 将秒转换为毫秒
execution_time_milliseconds = execution_time_seconds * 1000

print("程序执行时间：", execution_time_milliseconds, "毫秒")

# 定义图像和三维格式坐标轴
fig=plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

size = 1000
x0 = np.linspace(-10.0, 10.0, size)
x1 = np.linspace(-10.0, 10.0, size)

X, Y = np.meshgrid(x0, x1)

rlt = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        rlt[i, j] = testFunc.get_fun(np.array([[X[i, j]], [Y[i, j]]]))

ax1.plot_surface(X, Y, rlt, cmap='viridis', alpha=0.5)

x_tmp = np.array(x_tmp)

ax1.scatter(x_tmp[:, 0], x_tmp[:, 1], x_tmp[:, 2], c='r')

plt.show()

# testFunc = BFGS_func.Func()
#
# x = np.zeros((2, 1))
# x[0] = 2.2
# x[1] = -1.3
#
# print("initial x: ", x)
#
# x_tmp = []
#
# maxIter = 100000
# iter = 0
#
# start_time = time.time()
#
# g = testFunc.get_grad(x)
# d = -g
# t = weak_wolfe_conditions.get_weak_wolfe_step(x, d, testFunc)
# x_next = x + t * d
# g_next = testFunc.get_grad(x_next)
# # 为了提前存数据
#
# m = 5
# s = []
# y = []
# rho = []
#
# deltaX = x_next - x
# deltaG = g_next - g
#
# s.append(deltaX)
# y.append(deltaG)
# rho.append(1 / (deltaX.T @ deltaG)[0][0])
# x = x_next
# g = g_next
#
# while iter < maxIter:
#     currGrad = testFunc.get_grad(x)
#     norm_2 = np.linalg.norm(currGrad, ord=2)
#     x_tmp.append([x[0][0], x[1][0], testFunc.get_fun(x)])
#     if norm_2 < 1e-15:
#         print("iter: ",iter, "get min val: ", testFunc.get_fun(x))
#         print("opt x: ", x)
#         break
#     g = currGrad
#     d = LBFGS.get_L_BFGS_d(s, y, rho, g)
#     t = weak_wolfe_conditions.get_weak_wolfe_step(x, d, testFunc)
#     x_next = x + t * d
#     g_next = testFunc.get_grad(x_next)
#
#     deltaX = x_next - x
#     deltaG = g_next - g
#
#     tmp1 = (deltaG.T @ deltaX)[0][0]
#     tmp2 = 1e-6 * np.linalg.norm(g, ord=2) * (deltaX.T @ deltaX)[0][0]
#     if tmp1 > tmp2:
#         if len(s) > m:
#             del s[0]
#             del y[0]
#             del rho[0]
#
#         s.append(deltaX)
#         y.append(deltaG)
#         rho.append(1 / (deltaX.T @ deltaG)[0][0])
#
#
#     x = x_next
#
#     iter += 1
#
# # 记录结束时间
# end_time = time.time()
#
# # 计算执行时间（秒）
# execution_time_seconds = end_time - start_time
#
# # 将秒转换为毫秒
# execution_time_milliseconds = execution_time_seconds * 1000
#
# print("程序执行时间：", execution_time_milliseconds, "毫秒")
#
# # 定义图像和三维格式坐标轴
# fig=plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
#
# size = 1000
# x0 = np.linspace(-10.0, 10.0, size)
# x1 = np.linspace(-10.0, 10.0, size)
#
# X, Y = np.meshgrid(x0, x1)
#
# rlt = np.zeros_like(X)
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         rlt[i, j] = testFunc.get_fun(np.array([[X[i, j]], [Y[i, j]]]))
#
# ax1.plot_surface(X, Y, rlt, cmap='viridis', alpha=0.5)
#
# x_tmp = np.array(x_tmp)
#
# ax1.scatter(x_tmp[:, 0], x_tmp[:, 1], x_tmp[:, 2], c='r')
#
# plt.show()

