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

scriptName = "pointSource_normalLens_periodic"
symmetry = True # Impose symmetry around x = 0 line
periodic = True


def conic_filter2(x, radius, Lx, Ly, Nx, Ny):
    """A linear conic filter, also known as a "Hat" filter in the literature [1].
    Parameters
    ----------
    x : array_like (2D)
        Design parameters
    radius : float
        Filter radius (in "meep units")
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    Returns
    -------
    array_like (2D)
        Filtered design parameters.
    References
    ----------
    [1] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    """
    x = x.reshape(Nx, Ny)  # Ensure the input is 2D

    xv = np.arange(0, Lx / 2, Lx / Nx) # Lx / Nx instead of 1 / resolution
    yv = np.arange(0, Ly / 2, Ly / Ny)

    X, Y = np.meshgrid(xv, yv, sparse=True, indexing="ij")
    h = np.where(
        X**2 + Y**2 < radius**2, (1 - np.sqrt(abs(X**2 + Y**2)) / radius), 0
    )

    # Filter the response
    return mpa.simple_2d_filter(x, h)


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

# Dimensions
num_layers = 5 # amount of layers
design_region_width = 10 # width of layer
design_region_height = [0.080]*num_layers # height of layer
spacing = 0 # spacing between layers
half_total_height = sum(design_region_height) / 2 + (num_layers - 1) * spacing / 2
empty_space = 0 # free space in simulation left and right of layer

# Boundary conditions
pml_size = 1.0 # thickness of absorbing boundary layer
kpoint = mp.Vector3()
resolution = 50 # 50 --> amount of grid points per um; needs to be > 49 for TiOx and 0.55 um

# System size
Sx = design_region_width + 2 * empty_space
if not periodic:
    Sx += 2 * pml_size
# Sx = design_region_width + 2 * empty_space
Sy = 2 * pml_size + half_total_height * 2 + 14
cell_size = mp.Vector3(Sx, Sy)

# Frequencies
nf = 3 # Amount of frequencies studied
frequencies = 1./np.linspace(0.55, 0.65, nf)


# Feature size constraints
minimum_length = 0.09  # minimum length scale (microns)
eta_i = (
    0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
)
eta_e = 0.65  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution) # = int(resolution)

# Boundary conditions
pml_layers = [mp.PML(pml_size, direction=mp.Y) if periodic else mp.PML(pml_size)]

# Source
fcen = frequencies[1]
fwidth = 0.03 # 0.2
focal_point = 6
source_center = mp.Vector3(y=focal_point)
source_size = mp.Vector3(0, 0, 0) # Source covers width of lens
srcs = mp.GaussianSource(frequency=fcen, fwidth=fwidth)  # Gaussian source
source = mp.Source(srcs, component=mp.Ez, size=source_size, center=source_center)

# Amount of variables
Nx = int(design_region_resolution * design_region_width)
Ny = 1 # int(design_region_resolution * design_region_height)


design_variables = [mp.MaterialGrid(mp.Vector3(Nx), SiO2, TiOx, grid_type="U_MEAN") for i in range(num_layers)] # SiO2
design_regions = [mpa.DesignRegion(
    design_variables[i],
    volume=mp.Volume(
        center=mp.Vector3(y=-half_total_height + 0.5 * design_region_height[i] + sum(design_region_height[:i]) + i * spacing),
        size=mp.Vector3(design_region_width, design_region_height[i], 0),
    ),
) for i in range(num_layers)]


# Filter and projection
def mapping(x, eta, beta):

    # filter
    filtered_field = conic_filter2( # remain minimum feature size
        x,
        filter_radius,
        design_region_width,
        1, # design_region_height,
        Nx,
        Ny
    )

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta) # Make binary

    if symmetry:
        projected_field = (
            npa.flipud(projected_field) + projected_field
        ) / 2  # left-right symmetry

    # interpolate to actual materials
    return projected_field.flatten()

# Geometry: all is design region, no fixed parts

geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_region.design_parameters
    )
    # mp.Block(center=design_region.center, size=design_region.size, material=design_variables, e1=mp.Vector3(x=-1))
    #
    # The commented lines above impose symmetry by overlapping design region with the same design variable. However,
    # currently there is an issue of doing that; instead, we use an alternative approach to impose symmetry.
    # See https://github.com/NanoComp/meep/issues/1984 and https://github.com/NanoComp/meep/issues/2093
    for design_region in design_regions
    ]
geometry.append(mp.Block(
        center = mp.Vector3(y=-(half_total_height + Sy/2) / 2),
        size=mp.Vector3(x = Sx, y = (Sy/2 - half_total_height)),
        material=SiO2
    ))


sim = mp.Simulation(
		cell_size=cell_size,
		boundary_layers=pml_layers,
		geometry=geometry,
        k_point=kpoint if periodic else False,
		sources=source,
		default_material=Air, # Air
		symmetries=[mp.Mirror(direction=mp.X)] if symmetry else None,
		resolution=resolution,
	)


def J1(x):
    return
ob_list = [None]

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J1],
    objective_arguments=ob_list,
    design_regions=design_regions,
    frequencies=frequencies,
    maximum_run_time=2000,
)

file_path = "x.npy"
with open(file_path, 'rb') as file:
    x = np.load(file)
reshaped_x = np.reshape(x, [num_layers, Nx])

cur_beta = 256
# Plot final design
opt.update_design([mapping(reshaped_x[i, :], eta_i, cur_beta) for i in range(num_layers)])

plt.figure()
ax = plt.gca()
opt.plot2D(
    False,
    ax=ax,
    plot_sources_flag=False,
    plot_monitors_flag=False,
    plot_boundaries_flag=False,
)
circ = Circle((2, 2), minimum_length / 2)
ax.add_patch(circ)
ax.axis("off")
plt.savefig("./" + scriptName + "/design.png")

# Set-up simulation object

for freq in frequencies:
    srcs = mp.GaussianSource(frequency=freq, fwidth=fwidth)  # Gaussian source
    source = mp.Source(srcs, component=mp.Ez, size=source_size, center=source_center)
    opt.sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        k_point=kpoint if periodic else None,
        sources=[source],
        default_material=Air,  # Air
        symmetries=[mp.Mirror(direction=mp.X)] if symmetry else None,
        resolution=resolution)

    near_fields1 = opt.sim.add_dft_fields([mp.Ez], freq, 0, 1, center=mp.Vector3(y=-half_total_height - 0.1),
                                      size=mp.Vector3(x=Sx))
    near_fields2 = opt.sim.add_dft_fields([mp.Ez], freq, 0, 1, center=mp.Vector3(y=-Sy/2 + pml_size + 1),
                                      size=mp.Vector3(x=Sx))
    # print("start run")
    opt.sim.run(until=1000)
    plt.figure(figsize=(Sx, Sy))
    opt.sim.plot2D(fields=mp.Ez)
    fileName = "./" + scriptName + "/fieldAtWavelength" + str(1 / freq) + ".png"
    plt.savefig(fileName)

    plt.close()

    field1 = [opt.sim.get_dft_array(near_fields1, mp.Ez, 0)]
    field2 = [opt.sim.get_dft_array(near_fields2, mp.Ez, 0)]

    field1_modulus = np.abs(field1)
    field2_modulus = np.abs(field2)
    field1_angle = np.angle(field1)
    field2_angle = np.angle(field2)

    # [xi, yi, zi, wi] = opt.sim.get_array_metadata(dft_cell=field1)
    # [xj, yj, zj, wj] = opt.sim.get_array_metadata(dft_cell=field2)

    x1 = np.array(range(max(np.shape(field1_modulus)))) / max(np.shape(field1_modulus)) * Sx - Sx/2
    x1 = np.reshape(x1, [1, max(np.shape(field1_modulus))])
    x2 = x1

    # plot
    plt.figure()
    plt.plot(x1, field1_modulus, 'bo-')
    plt.xlabel("x (μm)")
    plt.ylabel("field modulus")
    fileName = f"./" + scriptName + "/modulus_after_lens_freq_" + str(1 / freq) + ".png"
    plt.savefig(fileName)

    plt.figure()
    plt.plot(x1, field1_angle, 'bo-')
    plt.xlabel("x (μm)")
    plt.ylabel("field angle")
    fileName = f"./" + scriptName + "/angle_after_lens_freq_" + str(1 / freq) + ".png"
    plt.savefig(fileName)

    plt.figure()
    plt.plot(x2, field2_modulus, 'bo-')
    plt.xlabel("x (μm)")
    plt.ylabel("field modulus")
    fileName = f"./" + scriptName + "/modulus_far_freq_" + str(1 / freq) + ".png"
    plt.savefig(fileName)

    plt.figure()
    plt.plot(x2, field2_angle, 'bo-')
    plt.xlabel("x (μm)")
    plt.ylabel("field angle")
    fileName = f"./" + scriptName + "/angle_far_freq_" + str(1 / freq) + ".png"
    plt.savefig(fileName)

