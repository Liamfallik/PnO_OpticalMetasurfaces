import os
import meep as mp
import meep.adjoint as mpa
from meep import Animate2D
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

import random
import datetime
import requests  # send notifications

scriptName = "metalens_3d_1layer_direct_design_nosim"
num_layers = 1

def focussing_efficiency(intensity1, intensity2):
    total_power = sum(intensity2)

    center = np.argmax(intensity1)
    length = max(np.shape(intensity1))

    value = intensity1[center]
    go = True
    i = center + 1
    while go and i < length:
        new_value = intensity1[i]
        if new_value > value:
            zero2 = i
            go = False
        else:
            value = new_value
        i += 1

    value = intensity1[center]
    go = True
    i = center - 1
    while go and i >= 0:
        new_value = intensity1[i]
        if new_value > value:
            zero1 = i
            go = False
        else:
            value = new_value
        i -= 1

    focussed_power = sum(intensity1[zero1+1:zero2])

    return focussed_power / total_power


def get_FWHM(intensity, x):

    center = np.argmax(intensity)
    length = max(np.shape(intensity))

    value = intensity[center]
    half_max = value / 2
    go = True
    i = center + 1
    while go and i < length:
        new_value = intensity[i]
        if new_value < half_max:
            half_right = x[i-1] + (x[i] - x[i-1]) * (value - half_max) / (value - new_value)
            go = False
        else:
            value = new_value
        i += 1

    value = intensity[center]
    go = True
    i = center - 1
    while go and i >= 0:
        new_value = intensity[i]
        if new_value < half_max:
            half_left = x[i+1] + (x[i] - x[i+1]) * (value - half_max) / (value - new_value)
            go = False
        else:
            value = new_value
        i -= 1

    return half_right - half_left


# checking if the directory demo_folder
# exist or not.
if not os.path.exists("./" + scriptName):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs("./" + scriptName)

# Dimensions
design_region_width = 10
design_region_height = [0.48 / (num_layers + 1)]*num_layers
half_total_height = sum(design_region_height) / 2

pml_size = 0.3
# System size
Sx = 2 * pml_size + design_region_width
Sz = 2 * pml_size + half_total_height * 2 + 1

Sz2 = 16

file_path = "./" + scriptName + "/intensity_at_focus_line.npy"
with open(file_path, 'rb') as file:
    focussed_amplitude_line = np.load(file)

file_path = "./" + scriptName + "/intensity_at_focus.npy"
with open(file_path, 'rb') as file:
    focussed_amplitude = np.load(file)

file_path = "./" + scriptName + "/intensity_before lens.npy"
with open(file_path, 'rb') as file:
    before_amplitude = np.load(file)

file_path = "./" + scriptName + "/intensity_XZ.npy"
with open(file_path, 'rb') as file:
    scattered_amplitude = np.load(file)

print(np.size(before_amplitude))
print(sum(sum(before_amplitude)) / np.size(before_amplitude))

# [xi, yi, zi, wi] = sim.get_array_metadata(dft_cell=near_fields_focus)
xi = np.linspace(-design_region_width / 2, design_region_width / 2, max(np.shape(focussed_amplitude)))
yi = xi
# [xk, yk, zk, wk] = sim.get_array_metadata(dft_cell=near_fields_focus_line)
xk = np.linspace(-design_region_width / 2, design_region_width / 2, max(np.shape(focussed_amplitude_line)))
# [xj, yj, zj, wj] = sim.get_array_metadata(dft_cell=near_fields)
xj = np.linspace(-Sx / 2, Sx / 2, np.shape(scattered_amplitude)[0])
zj = np.linspace(-Sz2 / 2, Sz2 / 2, np.shape(scattered_amplitude)[1])



# plot intensity XZ
print("start plotting...")
plt.figure(dpi=150)
plt.pcolormesh(xj, zj, np.rot90(np.rot90(np.rot90(scattered_amplitude))), cmap='inferno', shading='gouraud',
               vmin=0,
               vmax=scattered_amplitude.max())
plt.gca().set_aspect('equal')
plt.xlabel('x (μm)')
plt.ylabel('z (μm)')

# ensure that the height of the colobar matches that of the plot
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cax=cax)
plt.tight_layout()
fileName = f"./" + scriptName + "/intensityMapXZ" + ".png"
plt.savefig(fileName)
print("end plotting...")

# plot intensity XY focal
print("start plotting...")
plt.figure(dpi=150)
plt.pcolormesh(xi, yi, np.rot90(np.rot90(np.rot90(focussed_amplitude))), cmap='inferno', shading='gouraud',
               vmin=0,
               vmax=focussed_amplitude.max())
plt.gca().set_aspect('equal')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')

# ensure that the height of the colobar matches that of the plot
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cax=cax)
plt.tight_layout()
fileName = f"./" + scriptName + "/intensityFocusMapXY" + ".png"
plt.savefig(fileName)
print("end plotting...")

# plot intensity XY before
print("start plotting...")
plt.figure(dpi=150)
plt.pcolormesh(xi, yi, np.rot90(np.rot90(np.rot90(before_amplitude))), cmap='inferno', shading='gouraud',
               vmin=0,
               vmax=before_amplitude.max())
plt.gca().set_aspect('equal')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')

# ensure that the height of the colobar matches that of the plot
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cax=cax)
plt.tight_layout()
fileName = f"./" + scriptName + "/intensityMapXY" + ".png"
plt.savefig(fileName)
print("end plotting...")

# plot intensity around focal point
print("start plotting...")
plt.figure()
plt.plot(xk, focussed_amplitude_line, 'bo-')
plt.xlabel("x (μm)")
plt.ylabel("field amplitude")
fileName = f"./" + scriptName + "/intensityOverLine_atFocalPoint.png"
plt.savefig(fileName)
print("end plotting...")
# print(focussed_amplitude_line)
efficiency = focussing_efficiency(focussed_amplitude_line, focussed_amplitude_line)
FWHM = get_FWHM(focussed_amplitude_line, xk)


with open("./" + scriptName + "/best_result.txt", 'a') as var_file:
    var_file.write("focussing_efficiency \t" + str(efficiency) + "\n")
    var_file.write("FWHM \t" + str(FWHM) + "\n")

