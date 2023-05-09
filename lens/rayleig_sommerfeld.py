from math import pi
import numpy as np
from matplotlib import pyplot as plt
import random
import scipy.integrate as integ
import scipy

lamb = 0.6
f = 6
w = 10
k = 2*pi / lamb

# def integrate(g, bounds):
#     M = int(1e4)
#     dim = np.shape(bounds)[0]
#     value = 0
#     for i in range(M):
#         x = np.zeros(dim)
#         for j in range(dim):
#             x[j] = random.random() * (bounds[j][1] - bounds[j][0]) + bounds[j][0]
#         value += g(x)
#
#     area = 1
#     for j in range(dim):
#         area *= (bounds[j][1] - bounds[j][0])
#
#     return value / M * area


# 2D
g = lambda x, y, x2: -1j*k/(2*pi) * f * np.exp(1j*k * (np.sqrt(f**2 + (x-x2)**2 + (y)**2) - np.sqrt(f**2 + x**2))) / (f**2 + (x-x2)**2 + (y)**2)

real_g = lambda x, y, x2: np.real(g(x, y, x2))
imag_g = lambda x, y, x2: np.imag(g(x, y, x2))

N = 50
xi = np.linspace(-1, 1, N)
U2 = np.ones(N) * 1j
# yi = np.linspace(-2, 2, 0.001)

for i in range(N):
    # U2[i] = integrate(lambda x: g(x[0], x[1], xi[i]), [[-w/2, w/2], [-10 + xi[i], 10 + xi[i]]])
    U2[i] = integ.dblquad(lambda y, x: real_g(x, y, xi[i]), -w/2, w/2, -5, 5)[0] \
          + integ.dblquad(lambda y, x: imag_g(x, y, xi[i]), -w/2, w/2, -5, 5)[0]
    if (10 * i / N) % 1 == 0:
        print(str(int(100 * i / N) + 10) + " %")

plt.figure()
plt.plot(xi, np.abs(U2)**2)
plt.xlabel("x_2")
plt.ylabel("Intensity relative to source")
# plt.show()

# 3D
h = lambda x, y, x2, y2: -1j*k/(2*pi) * f * np.exp(1j*k * (np.sqrt(f**2 + (x-x2)**2 + (y - 0*y2)**2) -
                                                           np.sqrt(f**2 + x**2 + y**2))) / (f**2 + (x-x2)**2 + (y - y2)**2)

real_h = lambda x, y, x2, y2: np.real(h(x, y, x2, y2))
imag_h = lambda x, y, x2, y2: np.imag(h(x, y, x2, y2))

N = 50
xi = np.linspace(-1, 1, N)
U2 = np.zeros(N)
# yi = np.linspace(-2, 2, 0.001)

print(integ.dblquad(lambda y, x: real_h(x, y, 0, 0), -w/2, w/2, lambda x: -w/2, lambda x: w/2)[0] +
      integ.dblquad(lambda y, x: imag_h(x, y, 0, 0), -w/2, w/2, lambda x: -w/2, lambda x: w/2)[0])

for i in range(N):
    # U2[i] = integrate(lambda x: h(x[0], x[1], xi[i], 0), [[-w/2, w/2], [-w/2, w/2]])
    # print(integ.dblquad(lambda y, x: h(x, y, xi[i], 0), -w/2, w/2, lambda x: -w/2, lambda x: w/2))
    U2[i] = integ.dblquad(lambda y, x: real_h(x, y, xi[i], 0), -w/2, w/2, lambda x: -w/2, lambda x: w/2)[0] \
          + integ.dblquad(lambda y, x: imag_h(x, y, xi[i], 0), -w/2, w/2, lambda x: -w/2, lambda x: w/2)[0]
    if (10 * i / N) % 1 == 0:
        print(str(int(100 * i / N) + 10) + " %")

plt.figure()
plt.plot(xi, np.abs(U2)**2)
plt.xlabel("x_2")
plt.ylabel("Intensity relative to source")
plt.show()

