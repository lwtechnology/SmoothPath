import BFGS_func
import lewis_overton_line_search_func
import BFGS
import LBFGS
import lewis_overton_line_search
from matplotlib import pyplot as plt
import numpy as np
import time

# testFunc = lewis_overton_line_search_func.Func()
#
# x = np.zeros((2, 1))
# x[0] = 7.9
# x[1] = -0.5
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
# B = np.eye(2)
# val_prev = float('inf')
# while iter < maxIter:
#     currGrad = testFunc.get_grad(x)
#     norm_2 = np.linalg.norm(currGrad, ord=2)
#     val = testFunc.get_fun(x)
#     x_tmp.append([x[0][0], x[1][0], testFunc.get_fun(x)])
#     if abs(val - val_prev) < 1e-15:
#         print("iter: ",iter, "get min val: ", testFunc.get_fun(x))
#         print("opt x: ", x)
#         break
#     d = -B @ g
#     t = lewis_overton_line_search.get_lewis_overton_line_search_step(x, d, testFunc)
#     x_next = x + t * d
#     g_next = testFunc.get_grad(x_next)
#     B_next = BFGS.get_Cautious_BFGS_B(B, g_next - g, x_next - x, g)
#     x = x_next
#     g = g_next
#     B = B_next
#     val_prev = val
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

testFunc = lewis_overton_line_search_func.Func()

x = np.zeros((2, 1))
x[0] = 2.2
x[1] = -1.3

print("initial x: ", x)

x_tmp = []

maxIter = 100000
iter = 0

start_time = time.time()

g = testFunc.get_grad(x)
d = -g
t = lewis_overton_line_search.get_lewis_overton_line_search_step(x, d, testFunc)
x_next = x + t * d
g_next = testFunc.get_grad(x_next)
# 为了提前存数据

m = 5
s = []
y = []
rho = []

deltaX = x_next - x
deltaG = g_next - g

s.append(deltaX)
y.append(deltaG)
rho.append(1 / (deltaX.T @ deltaG)[0][0])
x = x_next
g = g_next

val_prev = float('inf')

while iter < maxIter:
    currGrad = testFunc.get_grad(x)
    norm_2 = np.linalg.norm(currGrad, ord=2)
    val = testFunc.get_fun(x)
    x_tmp.append([x[0][0], x[1][0], testFunc.get_fun(x)])
    if abs(val - val_prev) < 1e-15:
        print("iter: ",iter, "get min val: ", testFunc.get_fun(x))
        print("opt x: ", x)
        break
    g = currGrad
    d = LBFGS.get_L_BFGS_d(s, y, rho, g)
    t = lewis_overton_line_search.get_lewis_overton_line_search_step(x, d, testFunc)
    x_next = x + t * d
    g_next = testFunc.get_grad(x_next)

    deltaX = x_next - x
    deltaG = g_next - g

    tmp1 = (deltaG.T @ deltaX)[0][0]
    tmp2 = 1e-6 * np.linalg.norm(g, ord=2) * (deltaX.T @ deltaX)[0][0]
    if tmp1 > tmp2:
        if len(s) > m:
            del s[0]
            del y[0]
            del rho[0]

        s.append(deltaX)
        y.append(deltaG)
        rho.append(1 / (deltaX.T @ deltaG)[0][0])


    x = x_next

    val_prev = val

    iter += 1

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

