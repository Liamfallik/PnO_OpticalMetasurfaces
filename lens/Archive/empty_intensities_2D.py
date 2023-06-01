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
from scipy import special, signal
import datetime
import requests # send notifications
import random
from math import pi

start0 = datetime.datetime.now()
scriptName = "no_lens_2D_SiO2toAir4_fwidth_1"
symmetry = True # Impose symmetry around x = 0 line

# Dimensions
num_layers = 1 # amount of layers
design_region_width = 10 # width of layer
design_region_height = [0.24]*num_layers # height of layer
spacing = 0 # spacing between layers
half_total_height = sum(design_region_height) / 2 + (num_layers - 1) * spacing / 2
empty_space = 0 # free space in simulation left and right of layer

num_samples = 1 # 30


# checking if the directory demo_folder
# exist or not.
if not os.path.exists("./" + scriptName):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs("./" + scriptName)

mp.verbosity(1) # amount of info printed during simulation

# Materials
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.45)
NbOx = mp.Medium(index=2.5)
TiOx = mp.Medium(index=2.7) # 550 nm / 2.7 = 204 nm --> 20.4 nm resolution = 49
Air = mp.Medium(index=1.0)

# Boundary conditions
pml_size = 1 # thickness of absorbing boundary layer
resolution = 50 # 50 --> amount of grid points per µm; needs to be > 49 for TiOx and 0.55 µm

# System size
Sx = 2 * pml_size + design_region_width + 2 * empty_space
Sy = 2 * pml_size + half_total_height * 2 + 1
cell_size = mp.Vector3(Sx, Sy)

# Frequencies
nf = 3 # Amount of frequencies studied
frequencies = 1./np.linspace(0.55, 0.65, nf)
design_region_resolution = int(resolution) # = int(resolution)

# Boundary conditions
pml_layers = [mp.PML(pml_size)]

# Source
fcen = frequencies[nf // 2]
fwidth = 1 # 0.2
source_pos = -(half_total_height + 0.3)
source_center = [0, source_pos, 0] # Source 0.4 µm below lens
source_size = mp.Vector3(design_region_width + 2*empty_space, 0) # Source covers width of lens
# src = mp.GaussianSource(frequency=fcen, fwidth=fwidth) # Gaussian source
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
srcs = [mp.GaussianSource(frequency=fcen, fwidth=fwidth) for fcen in frequencies] # Gaussian source
source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center) for src in srcs]

# Amount of variables
Nx = int(design_region_resolution * design_region_width)
Ny = 1 # int(design_region_resolution * design_region_height)


# Geometry: all is design region, no fixed parts
# geometry = [mp.Block(
#         center = mp.Vector3(z=(half_total_height + Sy/2) / 2),
#         size=mp.Vector3(x = Sx, y  = Sx, z = (Sy/2 - half_total_height)),
#         material=SiO2
#     )]
geometry = [mp.Block(
        center = mp.Vector3(y=-Sy/4),
        size=mp.Vector3(x = Sx, y = Sy/2),
        material=SiO2
    )]
# geometry = [mp.Block(
#         center = mp.Vector3(),
#         size=mp.Vector3(x = Sx, y = 2*half_total_height),
#         material=TiOx
#     )]
# geometry = []

# Sy2 = 20
# geometry.append(mp.Block(
#     center=mp.Vector3(z=-(Sy2 / 2 + Sy / 2) / 2),
#     size=mp.Vector3(x=Sx, z=(Sy2 / 2 - Sy / 2)),
#     material=SiO2
# ))



focal_point = 6

for freq in frequencies:
    # simulate intensities

    src = mp.GaussianSource(frequency=freq, fwidth=fwidth)
    source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
    sim = mp.Simulation(resolution=resolution,
                        cell_size=mp.Vector3(Sx, Sy),
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=source,
                        symmetries=[mp.Mirror(direction=mp.X)] if symmetry else None,
                        )

    near_fields_before_line = sim.add_dft_fields([mp.Ez], freq, 0, 1, center=mp.Vector3(y=(-half_total_height + source_pos) / 2),
                                                size=mp.Vector3(x=design_region_width))
    near_fields_after_line = sim.add_dft_fields([mp.Ez], freq, 0, 1,
                                                center=mp.Vector3(y=-(-half_total_height + source_pos) / 2),
                                                 size=mp.Vector3(x=design_region_width))
    near_fields_behind_line = sim.add_dft_fields([mp.Ez], freq, 0, 1,
                                                center=mp.Vector3(y=(source_pos - 0.1)),
                                                 size=mp.Vector3(x=design_region_width))
    near_fields = sim.add_dft_fields([mp.Ez], freq, 0, 1, center=mp.Vector3(),
                                         size=mp.Vector3(x=Sx, y=Sy))


    sim.run(until=500)
    print(1)

    after_field_line = sim.get_dft_array(near_fields_after_line, mp.Ez, 0)
    before_field_line = sim.get_dft_array(near_fields_before_line, mp.Ez, 0)
    behind_field_line = sim.get_dft_array(near_fields_behind_line, mp.Ez, 0)
    scattered_field = sim.get_dft_array(near_fields, mp.Ez, 0)

    after_amplitude_line = np.abs(after_field_line) ** 2
    before_amplitude_line = np.abs(before_field_line) ** 2
    behind_amplitude_line = np.abs(behind_field_line) ** 2
    scattered_amplitude = np.abs(scattered_field) ** 2

    print(before_amplitude_line)
    print(after_amplitude_line)
    print(np.shape(before_amplitude_line))
    average_before = sum(before_amplitude_line) / np.size(before_amplitude_line)
    average_after = sum(after_amplitude_line) / np.size(after_amplitude_line)
    average_behind = sum(behind_amplitude_line) / np.size(behind_amplitude_line)
    print(average_before)
    print(average_after)

    [xi, yi, zi, wi] = sim.get_array_metadata(dft_cell=near_fields_after_line)
    [xj, yj, zj, wj] = sim.get_array_metadata(dft_cell=near_fields)

    with open("./" + scriptName + "/results.txt", 'w') as var_file:
        var_file.write("before \t" + str(average_before) + "\n")
        var_file.write("after \t" + str(average_after) + "\n")
        var_file.write("behind \t" + str(average_behind) + "\n")


    plt.figure()
    plt.plot(xi, before_amplitude_line, 'bo-')
    plt.xlabel("x (μm)")
    plt.ylabel("field amplitude")
    fileName = f"./" + scriptName + "/intensityOverLine_before.png"
    plt.savefig(fileName)

    plt.figure()
    plt.plot(xi, after_amplitude_line, 'bo-')
    plt.xlabel("x (μm)")
    plt.ylabel("field amplitude")
    fileName = f"./" + scriptName + "/intensityOverLine_after.png"
    plt.savefig(fileName)

    plt.figure()
    plt.plot(xi, behind_amplitude_line, 'bo-')
    plt.xlabel("x (μm)")
    plt.ylabel("field amplitude")
    fileName = f"./" + scriptName + "/intensityOverLine_behind.png"
    plt.savefig(fileName)

    # plot colormesh
    print(1)
    plt.figure(dpi=150)
    print(2)
    plt.pcolormesh(xj, yj, np.rot90(np.rot90(np.rot90(scattered_amplitude))), cmap='inferno', shading='gouraud',
                   vmin=0,
                   vmax=scattered_amplitude.max())
    print(3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    print(4)

    # ensure that the height of the colobar matches that of the plot
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    plt.tight_layout()
    print(5)
    fileName = f"./" + scriptName + "/intensityMapAllAfterAtWavelength" + str(
        int(100 / freq) / 100) + ".png"
    plt.savefig(fileName)
    print(6)

    plt.figure()
    sim.plot2D(fields=mp.Ez)
    fileName = f"./" + scriptName + "/fields_wavelength" + str(
        int(100 / freq) / 100) + ".png"
    plt.savefig(fileName)



    exit()

