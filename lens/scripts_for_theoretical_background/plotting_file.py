"""
Some plots for the formulas found in the report.
"""

import numpy as np
from math import pi, sqrt, log, cos
from matplotlib import pyplot as plt
import scipy.integrate as integ
f = 6
w = 10
lamb = 0.6

num_layers = np.array(range(6)) + 1
plt.rc('font', size=20)
plt.figure()
intensity_uniform = []
intensity_exp = []
for i in num_layers:
    intensity_uniform.append(20.99*(integ.quad(cos, 0, pi/(float(i)+1.0))[0]  / (pi/(float(i)+1.0)))**2)
    intensity_exp.append(20.99*(integ.quad(cos, 0, pi/2**float(i))[0]  / (pi/2**float(i)))**2)

plt.scatter(num_layers, intensity_uniform)
plt.scatter(num_layers, intensity_exp)
plt.plot(num_layers, [20.99]*num_layers[-1], '--')

plt.ylim([0, 23])
plt.show()
exit()

a = lambda x: (w/(2*x))**2
b = lambda x: 1/a(x)

k = 2*pi/lamb

I = lambda x: k**2*x**2/4 * (np.log(1 + a(x)) + pi**2/48/(1 + b(x)) + (1 + 4*b(x)) / (1 + b(x))**2 * pi**4 / 7680)**2
I_circ = lambda x: k**2*x**2/4 * np.log(1 + a(x))**2

print(I(f))

x = np.linspace(1/1000, 15, 1000)
Ix = I(x)
Ix_circ = I_circ(x)
plt.rc('font', size=20)
plt.figure()
plt.plot(x, Ix)
plt.plot(x, Ix_circ, '-.')
plt.legend(["rectangular lens", "circular lens"])
plt.xlabel("f [µm]")
plt.ylabel("Intensity at focal point")

print(np.max(Ix))
print(np.argmax(Ix))
print(x[np.argmax(Ix)])
print(Ix[:10])

integrand = lambda theta: 2*k*f/pi * np.log(1 + a(f) / np.cos(theta)**2)
zer_order = lambda theta: 2*k*f/pi * np.log(1 + a(f) ) + 0*theta
sec_order = lambda theta: 2*k*f/pi * (np.log(1 + a(f) ) + theta**2 / (1 + b(f)))
fou_order = lambda theta: 2*k*f/pi * (np.log(1 + a(f) ) + theta**2 / (1 + b(f)) + theta**4 / 6 * (1 + 4*b(f)) / (1 + b(f))**2)
theta = np.linspace(0, pi/4, 40)
plt.rc('font', size=15)
plt.figure()
plt.plot(theta, integrand(theta))
plt.plot(theta, zer_order(theta), 'x')
plt.plot(theta, sec_order(theta), '-.')
plt.plot(theta, fou_order(theta), '--')
plt.legend(["exact", "zeroth order (circular lens)", "secondth order", "fourth order"])
plt.xlabel("theta [rad]")
plt.ylabel("integrand")
plt.ylim([0, 36])
plt.show()