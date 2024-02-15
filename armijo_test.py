import armijo_func
from matplotlib import pyplot as plt
import Armijo
import numpy as np

testFunc = armijo_func.Func()

# x = np.random.uniform(-10, 10, size=2)
x = np.array([9.5, -8.8])

print("initial x: ", x)

x_tmp = []

maxIter = 100000
iter = 0
while iter < maxIter:
    currGrad = testFunc.get_grad(x)
    norm_2 = np.linalg.norm(currGrad, ord=2)
    x_tmp.append([x[0], x[1], testFunc.get_fun(x)])
    if norm_2 < 1e-8:
        print("iter: ",iter, "get min val: ", testFunc.get_fun(x))
        print("opt x: ", x)
        break
    d = -currGrad
    step = Armijo.get_armijo_step(x, d, testFunc)
    x = x + step * d
    iter += 1

# 定义图像和三维格式坐标轴
fig=plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

size = 1000
x0 = np.linspace(-10, 10, size)
x1 = np.linspace(-10, 10, size)

X, Y = np.meshgrid(x0, x1)

rlt = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        rlt[i, j] = testFunc.get_fun(np.array([X[i, j], Y[i, j]]))

ax1.plot_surface(X, Y, rlt, cmap='viridis', alpha=0.5)

x_tmp = np.array(x_tmp)

ax1.scatter(x_tmp[:, 0], x_tmp[:, 1], x_tmp[:, 2], c='r')

plt.show()