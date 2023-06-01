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
scriptName = "no_lens_3D_AirtoSiO2_fwidth_1"
symmetry = True # Impose symmetry around x = 0 line

# Dimensions
num_layers = 1 # amount of layers
design_region_width = 3 # width of layer
design_region_height = [0.01]*num_layers # height of layer
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
pml_size = 0.3 # thickness of absorbing boundary layer
resolution = 50 # 50 --> amount of grid points per µm; needs to be > 49 for TiOx and 0.55 µm

# System size
Sx = 2 * pml_size + design_region_width + 2 * empty_space
Sy = 2 * pml_size + half_total_height * 2 + 1
cell_size = mp.Vector3(Sx, Sx, Sy)

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
source_center = [0, 0, source_pos] # Source 0.4 µm below lens
source_size = mp.Vector3(design_region_width + 2*empty_space, design_region_width + 2*empty_space, 0) # Source covers width of lens
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
        center = mp.Vector3(z=Sy/4),
        size=mp.Vector3(x = Sx, y  = Sx, z = Sy/2),
        material=SiO2
    )]

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
    source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]
    sim = mp.Simulation(resolution=resolution,
                        cell_size=mp.Vector3(Sx, Sx, Sy),
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=source,
                        symmetries=[mp.Mirror(direction=mp.X, phase=-1),
                                    mp.Mirror(direction=mp.Y)] if symmetry else None,
                        )


    near_fields_before = sim.add_dft_fields([mp.Ex], freq, 0, 1, center=mp.Vector3(z=(-half_total_height + source_pos) / 2),
                                                size=mp.Vector3(x=design_region_width, y=design_region_width))
    near_fields_before_line = sim.add_dft_fields([mp.Ex], freq, 0, 1, center=mp.Vector3(z=(-half_total_height + source_pos) / 2),
                                                size=mp.Vector3(y=design_region_width))
    near_fields_after = sim.add_dft_fields([mp.Ex], freq, 0, 1,
                                                center=mp.Vector3(z=-(-half_total_height + source_pos) / 2),
                                                 size=mp.Vector3(x=design_region_width, y=design_region_width))
    near_fields_after_line = sim.add_dft_fields([mp.Ex], freq, 0, 1,
                                                center=mp.Vector3(z=-(-half_total_height + source_pos) / 2),
                                                 size=mp.Vector3(y=design_region_width))
    near_fields = sim.add_dft_fields([mp.Ex], freq, 0, 1, center=mp.Vector3(),
                                         size=mp.Vector3(y=Sx, z=Sy))


    sim.run(until=200)
    print(1)

    after_field = sim.get_dft_array(near_fields_after, mp.Ex, 0)
    before_field = sim.get_dft_array(near_fields_before, mp.Ex, 0)
    after_field_line = sim.get_dft_array(near_fields_after_line, mp.Ex, 0)
    before_field_line = sim.get_dft_array(near_fields_before_line, mp.Ex, 0)
    scattered_field = sim.get_dft_array(near_fields, mp.Ex, 0)

    after_amplitude = np.abs(after_field) ** 2
    scattered_amplitude = np.abs(scattered_field) ** 2
    after_amplitude_line = np.abs(after_field_line) ** 2
    before_amplitude_line = np.abs(before_field_line) ** 2
    before_amplitude = np.abs(before_field) ** 2

    print(before_amplitude)
    print(after_amplitude)
    print(np.shape(before_amplitude))
    average_before = sum(sum(before_amplitude)) / np.size(before_amplitude)
    average_after = sum(sum(after_amplitude)) / np.size(after_amplitude)
    print(average_before)
    print(average_after)

    with open("./" + scriptName + "/results.txt", 'w') as var_file:
        var_file.write("before \t" + str(average_before) + "\n")
        var_file.write("after \t" + str(average_after) + "\n")

    [xi, yi, zi, wi] = sim.get_array_metadata(dft_cell=near_fields_after)
    [xj, yj, zj, wj] = sim.get_array_metadata(dft_cell=near_fields)

    # plot colormesh
    print(1)
    plt.figure(dpi=150)
    print(2)
    plt.pcolormesh(xi, yi, np.rot90(np.rot90(np.rot90(before_amplitude))), cmap='inferno', shading='gouraud',
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
    fileName = f"./" + scriptName + "/intensityMapBeforeAtWavelength" + str(
        int(100 / freq) / 100) + ".png"
    plt.savefig(fileName)
    print(6)


    # plot colormesh
    print(1)
    plt.figure(dpi=150)
    print(2)
    plt.pcolormesh(xi, yi, np.rot90(np.rot90(np.rot90(after_amplitude))), cmap='inferno', shading='gouraud',
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
    fileName = f"./" + scriptName + "/intensityMapAfterAtWavelength" + str(
        int(100 / freq) / 100) + ".png"
    plt.savefig(fileName)
    print(6)

    # plot colormesh
    print(1)
    plt.figure(dpi=150)
    print(2)
    plt.pcolormesh(yj, zj, np.rot90(np.rot90(np.rot90(scattered_amplitude))), cmap='inferno', shading='gouraud',
                   vmin=0,
                   vmax=scattered_amplitude.max())
    print(3)
    plt.gca().set_aspect('equal')
    plt.xlabel('y (μm)')
    plt.ylabel('z (μm)')
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
    plt.plot(xi, before_amplitude_line, 'bo-')
    plt.xlabel("y (μm)")
    plt.ylabel("field amplitude")
    fileName = f"./" + scriptName + "/intensityOverLine_before.png"
    plt.savefig(fileName)

    plt.figure()
    plt.plot(xi, after_amplitude_line, 'bo-')
    plt.xlabel("y (μm)")
    plt.ylabel("field amplitude")
    fileName = f"./" + scriptName + "/intensityOverLine_after.png"
    plt.savefig(fileName)

    exit()

