"""
Plots the required frequencies, as discussed in the report.
"""

import numpy as np
import random
from matplotlib import pyplot as plt
import os

frequency = 1/0.6
focal_point = 6
phase0 = 0.1
width = 10
resolution = 100

Nx = width * resolution
phase = np.zeros(Nx)
x = np.zeros(Nx)

for j in range(Nx//2, Nx):
    x[j] = (j - Nx//2) / resolution
    phase[j] = (phase0 - np.sqrt(focal_point**2 + (x[j])**2) * frequency) % 1
    x[Nx - 1 - j] = -x[j]
    phase[Nx - 1 - j] = phase[j]

plt.figure()
plt.plot(x, phase)
plt.xlabel("x [µm]")
plt.ylabel("phase [/360°]")
plt.xlim((x[0], x[-1]))

n = 1.45
frequencies = [1/0.47 * n, 1/0.65 * n]
nf = len(frequencies)
focal_point = 6
off_sets = [-3, 3]
phase0 = [0.1, 0]
width = 10
resolution = 100

Nx = width * resolution
phase = np.zeros([nf, Nx])
x = np.zeros(Nx)

for j in range(Nx):
    x[j] = (j - Nx//2) / resolution
    for i in range(nf):
        phase[i, j] = (phase0[i] - np.sqrt(focal_point**2 + (x[j] - off_sets[i])**2) * frequencies[i] * 1.45) % 1


plt.figure()
plt.plot(x, phase[0, :])
plt.plot(x, phase[1, :], "-.")
plt.xlabel("x [µm]")
plt.ylabel("phase [/360°]")
plt.xlim((x[0], x[-1]))
plt.legend(["blue", "red"])

plt.figure()
plt.scatter(phase[0, :], phase[1, :], 2)
plt.xlabel("blue phase [/360°]")
plt.ylabel("red phase [/360°]")
# plt.xlim((x[0], x[-1]))
plt.show()