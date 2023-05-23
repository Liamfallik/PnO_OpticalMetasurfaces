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
scriptName = "Retroreflector_intensities" # 20 deg and 10 deg
symmetry = False # Impose symmetry around x = 0 line


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

    if True:
        filtered_field = (
            npa.flipud(filtered_field) + filtered_field
        ) / 2  # left-right symmetry

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)  # Make binary

    # interpolate to actual materials
    return projected_field.flatten()

# checking if the directory demo_folder
# exist or not.
if not os.path.exists("./" + scriptName):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs("./" + scriptName)

mp.verbosity(0) # amount of info printed during simulation

# Materials
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.45)
NbOx = mp.Medium(index=2.5)
TiOx = mp.Medium(index=2.7) # 550 nm / 2.7 = 204 nm --> 20.4 nm resolution = 49
Air = mp.Medium(index=1.0)

# Dimensions
num_layers = 2 # amount of layers
design_region_width = 10 # width of layer
design_region_height = [0.24,0.24/2] # height of layer
spacing = 4 # spacing between layers
half_total_height = sum(design_region_height) / 2 + (num_layers - 1) * spacing / 2
empty_space = 0 # free space in simulation left and right of layer

# Boundary conditions
pml_size = 1.0 # thickness of absorbing boundary layer

resolution = 50 # 50 --> amount of grid points per µm; needs to be > 49 for TiOx and 0.55 µm

# System size
space_below = 1.7 # includes PML
Sx = 2 * pml_size + design_region_width + 2 * empty_space
Sy = half_total_height * 2 + space_below
cell_size = mp.Vector3(Sx, Sy)



# Frequencies
nf = 1 # Amount of frequencies studied
frequencies = 1./np.linspace(0.55, 0.65, nf)

# Feature size constraints
minimum_length = 0.09  # minimum length scale (microns)
eta_i = (
    0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution) # = int(resolution)

# Boundary conditions
pml_layers = [mp.PML(pml_size, direction=mp.X), mp.PML(pml_size, direction=mp.Y, side=1)]

# Amount of variables
Nx = int(design_region_resolution * design_region_width)
Ny = 1 # int(design_region_resolution * design_region_height)

# Source
fcen = frequencies[0]
fwidth = 0.4 # 0.2
source_intensity = 0.04074 / fwidth**2
source_angles = [np.radians(0), np.radians(5), np.radians(10), np.radians(15), np.radians(20), np.radians(25)]
# source_angles = [np.radians(0), np.radians(10), np.radians(20)]

# rot_angle = np.radians(20)
kpoints = [mp.Vector3(y=1).rotate(mp.Vector3(z=1), -rot_angle) for rot_angle in source_angles]
source_center = [0, -(half_total_height - space_below / 2 + 0.4), 0] # Source 1 µm below lens
source_size = mp.Vector3(design_region_width, 0, 0) # Source covers width of lens
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth) # Gaussian source
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
# srcs = [mp.GaussianSource(frequency=fcen, fwidth=fwidth)] # Gaussian source
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center) for src in srcs]
sources = [mp.EigenModeSource(
        src,
        eig_band=1,
        direction=mp.NO_DIRECTION,
        eig_kpoint=kpoint,
        eig_parity=mp.ODD_Z,
        size=source_size,
        center=source_center,
    ) for kpoint in kpoints]

rot_angle2 = np.radians(10)
kpoint2 = mp.Vector3(y=1).rotate(mp.Vector3(z=1), -rot_angle2)
# source_center = [0, -(half_total_height - space_below / 2 + 0.4), 0] # Source 1 µm below lens
# source_size = mp.Vector3(design_region_width, 0, 0) # Source covers width of lens
# srcs = [mp.GaussianSource(frequency=fcen, fwidth=fwidth) for fcen in frequencies] # Gaussian source


design_variables = [mp.MaterialGrid(mp.Vector3(Nx), SiO2, TiOx, grid_type="U_MEAN") for i in range(num_layers)] # SiO2
design_regions = [mpa.DesignRegion(
    design_variables[i],
    volume=mp.Volume(
        center=mp.Vector3(y=-half_total_height + 0.5 * design_region_height[i] + sum(design_region_height[:i]) + i * spacing + space_below / 2),
        size=mp.Vector3(design_region_width, design_region_height[i], 0),
    ),
) for i in range(num_layers)]


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
        center = mp.Vector3(y=(Sy/2) - design_region_height[1] - (spacing/2)),
        size=mp.Vector3(x = Sx, y = spacing),
        material=SiO2
    ))

# # Set-up simulation object
# kpoint = mp.Vector3(1,1,0)

sim = [mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=[source],
    default_material=Air, # Air
    symmetries=[mp.Mirror(direction=mp.X)] if symmetry else None,
    resolution=resolution,
) for source in sources]


# Focal point 1
focal_point = -1000
n_angles = 500
angles_to_measure = np.linspace(-pi/4, pi/4, n_angles)
print(np.size(angles_to_measure))
far_x = [mp.Vector3(focal_point*np.sin(angle_to_measure), focal_point*np.cos(angle_to_measure), 0) for angle_to_measure in angles_to_measure]
NearRegions = [ # region from which fields at focus point will be calculated
    mp.Near2FarRegion(
        center=mp.Vector3(0, -(half_total_height - space_below / 2 + 0.5)), # 0.4 µm above lens
        size=mp.Vector3(design_region_width + 2*empty_space, 0), # spans design region
        weight= -1, # field contribution is positive (real)
    )
]
FarFields = [mpa.Near2FarFields(si, NearRegions, far_x) for si in sim] # Far-field object
ob_list = FarFields


def J1(FF):
    # print(FF)
    # # FF = FF[0, :, 2]
    # y_pos = -0.6
    # xi = np.linspace(-design_region_width/2 - y_pos / np.tan(rot_angle),
    #                            design_region_width/2 - y_pos / np.tan(rot_angle),
    #                            design_region_width * resolution*5)
    # FF = [sim.get_field_point(mp.Ez, mp.Vector3(x=i, y=-half_total_height+y_pos)) for i in xi]
    #
    # print(FF)
    #
    # ideal = np.exp(1j * 2*pi * np.sin(rot_angle) * frequencies * xi)
    #
    # mode_overlap = np.abs(sum(np.conjugate(FF) * ideal)) ** 2 / (sum(np.abs(FF)**2) * sum(np.abs(ideal)**2))

    # return mode_overlap
    return npa.mean(npa.abs(FF[0, :, 2]) ** 2) # only first (only point), mean of all frequencies, and third field (Ez)

# Initial guess
seed = 240 # make sure starting conditions are random, but always the same. Change seed to change starting conditions
np.random.seed(seed)


with open("./" + scriptName + "/used_variables.txt", 'w') as var_file:
    var_file.write("num_layers \t" + str(num_layers) + "\n")
    var_file.write("symmetry \t" + str(symmetry) + "\n")
    var_file.write("design_region_width \t" + str(design_region_width) + "\n")
    var_file.write("design_region_height \t" + str(design_region_height) + "\n")
    var_file.write("design_region_resolution \t" + str(design_region_resolution) + "\n")
    var_file.write("minimum_length \t" + str(minimum_length) + "\n")
    var_file.write("spacing \t" + str(spacing) + "\n")
    var_file.write("empty_space \t" + str(empty_space) + "\n")
    var_file.write("pml_size \t" + str(pml_size) + "\n")
    var_file.write("resolution \t" + str(resolution) + "\n")
    var_file.write("wavelengths \t" + str(1/frequencies) + "\n")
    var_file.write("fwidth \t" + str(fwidth) + "\n")
    var_file.write("focal_point \t" + str(focal_point) + "\n")
    var_file.write("seed \t%d" % seed + "\n")
file_path = "x.npy"
with open(file_path, 'rb') as file:
    x = np.load(file)

cur_beta = 4*2**6
reshaped_x = np.reshape(x, [num_layers, Nx])
mapped_x = [mapping(reshaped_x[i, :], eta_i, cur_beta) for i in range(num_layers)]

intensities = []
efficiency = []

for i in range(len(source_angles)):
    print("Incident angle = " + str(round(180/pi*source_angles[i])) + "°")
    # Optimization object
    opt = mpa.OptimizationProblem(
        simulation=sim[i],
        objective_functions=[J1],
        objective_arguments=[ob_list[i]],
        design_regions=design_regions,
        frequencies=frequencies,
        maximum_run_time=2000,
    )

    plt.figure()
    opt.plot2D(True)
    plt.savefig("./" + scriptName + "/optimizationDesign.png")



    opt.update_design(mapped_x)
    # opt2.update_design(mapped_x)

    # Check intensities in optimal design
    f01, dJ_du1 = opt([mapping(reshaped_x[i, :], eta_i, cur_beta // 2) for i in range(num_layers)], need_gradient=False)

    # frequencies = opt.frequencies
    intensities.append(np.abs(opt.get_objective_arguments()[0][:, 0, 2]) ** 2 / (source_intensity*design_region_width) * pi * abs(focal_point) / 180)

    plt.figure()
    plt.plot(angles_to_measure * 180 / pi, intensities[i])
    plt.plot([source_angles[i] * 180 / pi]*2, [0, max(intensities[i])], '-.')
    plt.xlabel("Reflectance angle [degrees]")
    plt.ylabel("Relative intensity per degree")

    fileName = f"./" + scriptName + "/intensityVsAngle_incident" + str(round(180/pi*source_angles[i])) + ".png"
    plt.savefig(fileName)

    plt.figure()
    zoomed_index_start = round((45 + source_angles[i]*180/pi - 5) / 90 * n_angles)
    zoomed_index_end = round((45 + source_angles[i]*180/pi + 5) / 90 * n_angles)
    plt.plot(angles_to_measure[zoomed_index_start:zoomed_index_end] * 180 / pi, intensities[i][zoomed_index_start:zoomed_index_end])
    plt.plot([source_angles[i] * 180 / pi]*2, [0, max(intensities[i])], '-.')
    plt.xlabel("Reflectance angle [degrees]")
    plt.ylabel("Relative intensity per degree")

    fileName = f"./" + scriptName + "/intensityVsAngle_incident" + str(round(180/pi*source_angles[i])) + "_zoomed" + ".png"
    plt.savefig(fileName)

    dtheta = 2
    index_start = round((45 + source_angles[i]*180/pi - dtheta) / 90 * n_angles)
    index_end = round((45 + source_angles[i]*180/pi + dtheta) / 90 * n_angles)
    efficiency.append(sum(intensities[i][index_start:index_end]) * 90 / n_angles)
    print(sum(intensities[i]) * 90 / n_angles)
    print(efficiency[i])


plt.figure()
for i in range(len(source_angles)):
    plt.plot(angles_to_measure * 180 / pi, intensities[i])
plt.xlabel("Reflectance angle [degrees]")
plt.ylabel("Relative intensity per degree")
plt.legend(["Incident: " + str(round(180/pi*angle)) + "°" for angle in source_angles], loc='upper left')
for i in range(len(source_angles)):
    plt.plot([source_angles[i] * 180 / pi]*2, [0, max(intensities[i])], '-.', color='k')
fileName = f"./" + scriptName + "/intensityVsAngle_together.png"
plt.savefig(fileName)

plt.figure()
plt.plot(np.array(source_angles) * 180/pi, np.array(efficiency)*100)
plt.xlabel("Incident angle [degrees]")
plt.ylabel("Efficiency [%]")
fileName = f"./" + scriptName + "/efficiency" + ".png"
plt.savefig(fileName)


with open("./" + scriptName + "/results.txt", 'w') as var_file:
    var_file.write("efficiencies \t" + str(efficiency) + "\n")