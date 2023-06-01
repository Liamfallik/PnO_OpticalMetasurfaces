"""
Rayleigh-Sommerfeld integrals for 2D and 3D lenses. Theory in report.
"""

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


# 2D


g = lambda x, y, x2: -1j*k/(2*pi) * f * np.exp(1j*k * (np.sqrt(f**2 + (x-x2)**2 + (y)**2) - np.sqrt(f**2 + x**2))) / (f**2 + (x-x2)**2 + (y)**2)
g2 = lambda x, y: -1j*k/(2*pi) * f * np.exp(1j*k * (np.sqrt(f**2 + x**2 + y**2) - np.sqrt(f**2 + x**2))) / (f**2 + x**2 + y**2)
g3 = lambda x, y, f: -1j*k/(2*pi) * f * np.exp(1j*k * (np.sqrt(f**2 + x**2 + y**2) - np.sqrt(f**2 + x**2))) / (f**2 + x**2 + y**2)

real_g = lambda x, y, x2: np.real(g(x, y, x2))
imag_g = lambda x, y, x2: np.imag(g(x, y, x2))
real_g2 = lambda x, y: np.real(g2(x, y))
imag_g2 = lambda x, y: np.imag(g2(x, y))
real_g3 = lambda x, y, z: np.real(g3(x, y, z))
imag_g3 = lambda x, y, z: np.imag(g3(x, y, z))
function_to_optimize = lambda z: 4*np.abs(integ.dblquad(lambda y, x: real_g3(x, y, z), 0, w / 2, 0, 30)[0] \
            + 1j * integ.dblquad(lambda y, x: imag_g3(x, y, z), 0, w / 2, 0, 30)[0])**(-2)

# opti = scipy.optimize.minimize_scalar(function_to_optimize, bounds=[2.2, 3])
# print(opti)

print(np.abs(4*integ.dblquad(lambda y, x: real_g2(x, y), 0, w/2, 0, 30)[0] \
    + 4*1j*integ.dblquad(lambda y, x: imag_g2(x, y), 0, w/2, 0, 30)[0]) ** 2)

N = 50
fs = np.linspace(1/100, 10, N)
U2 = np.ones(N) * 1j
for i in range(N):
    U2[i] = 4*integ.dblquad(lambda y, x: real_g3(x, y, fs[i]), 0, w / 2, 0, 30)[0] \
            + 1j * 4*integ.dblquad(lambda y, x: imag_g3(x, y, fs[i]), 0, w / 2, 0, 30)[0]
    if (10 * i / N) % 1 == 0:
        print(str(int(100 * i / N) + 10) + " %")


plt.rc('font', size=25)
plt.figure()
plt.plot(fs, np.abs(U2)**2)
plt.scatter([1, 2, 2.74, 4, 6], [30.109, 28.360, 27.056, 24.561, 20.509], c='red')
plt.xlabel("f [µm]")
plt.legend(["Theoretical", "Experimental"], loc='bottom right')
# plt.rcParams.update({'font.size': 100})
plt.rc('font', size=25)
plt.ylabel("Intensity at focal point")
plt.show()

N = 100
dx = 1.5
xi = np.linspace(-dx*0, dx, N)
U2 = np.ones(N) * 1j
# yi = np.linspace(-2, 2, 0.001)

for i in range(N):
    # U2[i] = integrate(lambda x: g(x[0], x[1], xi[i]), [[-w/2, w/2], [-10 + xi[i], 10 + xi[i]]])
    U2[i] = integ.dblquad(lambda y, x: real_g(x, y, xi[i]), -w/2, w/2, -10, 10)[0] \
          + 1j*integ.dblquad(lambda y, x: imag_g(x, y, xi[i]), -w/2, w/2, -10, 10)[0]
    if (10 * i / N) % 1 == 0:
        print(str(int(100 * i / N) + 10) + " %")

print(sum(np.abs(U2)**2) * 2*dx / np.size(U2) / w)

plt.rc('font', size=25)
plt.figure()
plt.plot(np.concatenate((-xi[-1:0:-1], xi)), np.abs(np.concatenate((U2[-1:0:-1], U2)))**2)
plt.xlabel("x_2")
plt.ylabel("Intensity relative to source")

# 3D
h = lambda x, y, x2, y2: -1j*k/(2*pi) * f * np.exp(1j*k * (np.sqrt(f**2 + (x-x2)**2 + (y - 0*y2)**2) -
                                                           np.sqrt(f**2 + x**2 + y**2))) / (f**2 + (x-x2)**2 + (y - y2)**2)

real_h = lambda x, y, x2, y2: np.real(h(x, y, x2, y2))
imag_h = lambda x, y, x2, y2: np.imag(h(x, y, x2, y2))

lnint = lambda x: 4*f/lamb * np.log(1 + (w/(2*f*np.cos(x)))**2)

print((integ.quad(lnint, 0, pi/4)[0])**2)

N = 100
xi = np.linspace(-dx*0, dx, N)
U2 = np.ones(N) * 1j
# yi = np.linspace(-2, 2, 0.001)

print(np.abs(integ.dblquad(lambda y, x: real_h(x, y, 0, 0), -w/2, w/2, lambda x: -w/2, lambda x: w/2)[0] +
      1j*integ.dblquad(lambda y, x: imag_h(x, y, 0, 0), -w/2, w/2, lambda x: -w/2, lambda x: w/2)[0]) ** 2)

for i in range(N):
    # U2[i] = integrate(lambda x: h(x[0], x[1], xi[i], 0), [[-w/2, w/2], [-w/2, w/2]])
    # print(integ.dblquad(lambda y, x: h(x, y, xi[i], 0), -w/2, w/2, lambda x: -w/2, lambda x: w/2))
    U2[i] = integ.dblquad(lambda y, x: real_h(x, y, xi[i], 0), -w/2, w/2, lambda x: -w/2, lambda x: w/2)[0] \
          + 1j*integ.dblquad(lambda y, x: imag_h(x, y, xi[i], 0), -w/2, w/2, lambda x: -w/2, lambda x: w/2)[0]
    if (10 * i / N) % 1 == 0:
        print(str(int(100 * i / N) + 10) + " %")

print(sum(np.abs(U2)**2) * 2*dx / np.size(U2) / w)

plt.rc('font', size=25)
plt.figure()
plt.plot(np.concatenate((-xi[-1:0:-1], xi)), np.abs(np.concatenate((U2[-1:0:-1], U2)))**2)
plt.xlabel("x_2")
plt.ylabel("Intensity relative to source")
plt.show()

