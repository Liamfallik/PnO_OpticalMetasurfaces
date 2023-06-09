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
import requests  # send notifications

scriptName = "metalens_1layer_2freq_img_test"


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

    xv = np.arange(0, Lx / 2, Lx / Nx)  # Lx / Nx instead of 1 / resolution
    yv = np.arange(0, Ly / 2, Ly / Ny)

    X, Y = np.meshgrid(xv, yv, sparse=True, indexing="ij")
    h = np.where(
        X ** 2 + Y ** 2 < radius ** 2, (1 - np.sqrt(abs(X ** 2 + Y ** 2)) / radius), 0
    )

    # Filter the response
    return mpa.simple_2d_filter(x, h)


# checking if the directory exists, else create it
if not os.path.exists("./" + scriptName):
    os.makedirs("./" + scriptName)

mp.verbosity(1)  # amount of info printed during simulation

# Materials
Si = mp.Medium(index=3.4)
Air = mp.Medium(index=1.0)

# Dimensions
design_region_width = 15
design_region_height = 2

# Boundary conditions
pml_size = 1.0

resolution = 30

# System size
Sx = 2 * pml_size + design_region_width
Sy = 2 * pml_size + design_region_height + 5
cell_size = mp.Vector3(Sx, Sy)

# Frequencies
f_red = 1 / 0.65
f_blue = 1 / 0.47
frequencies = [f_red, f_blue]

# Feature size constraints
minimum_length = 0.09  # minimum length scale (microns)
eta_i = (
    0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution)

# Boundary conditions
pml_layers = [mp.PML(pml_size)]

width = 0.05  # Relative width of frequency
source_center = [0, -(design_region_height / 2 + 1.5), 0]  # Source 1.5 µm below lens
source_size = mp.Vector3(design_region_width, 0, 0)  # Source covers width of lens
sources = [
    mp.Source(mp.GaussianSource(frequency=f_red, fwidth=width * f_red), component=mp.Ez, size=source_size,
              center=source_center),
    mp.Source(mp.GaussianSource(frequency=f_blue, fwidth=width * f_blue), component=mp.Ez, size=source_size,
              center=source_center)
]

# Amount of variables
Nx = int(design_region_resolution * design_region_width)
Ny = 1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), Air, Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(design_region_width, design_region_height, 0),
    ),
)


# Filter and projection
def mapping(x, eta, beta):
    # filter
    filtered_field = conic_filter2(  # remain minimum feature size
        x,
        filter_radius,
        design_region_width,
        design_region_height,
        Nx,
        Ny
    )

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)  # Make binary

    # projected_field = (
    #     npa.flipud(projected_field) + projected_field
    # ) / 2  # left-right symmetry

    # interpolate to actual materials
    return projected_field.flatten()


# Geometry: all is design region, no fixed parts
geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    )
]

# Set-up simulation object
sim = mp.Simulation(
    cell_size=mp.Vector3(Sx, Sy + 40),
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    default_material=Air,
    resolution=resolution,
)

# Focus points, 30 µm beyond centre of lens, separated by 10 µm
far_x = [mp.Vector3(-5, 30, 0), mp.Vector3(5, 30, 0)]
NearRegions = [  # region from which fields at focus point will be calculated
    mp.Near2FarRegion(
        center=mp.Vector3(0, design_region_height / 2 + 1.5),  # 1.5 µm above lens
        size=mp.Vector3(design_region_width, 0),  # spans design region
        weight=+1,  # field contribution is positive (real)
    )
]
FarFields = mpa.Near2FarFields(sim, NearRegions, far_x)  # Far-field object
ob_list = [FarFields]


def J1(FF):
    """
    Returns the minimum between the value for |Ez|² for both frequencies at their respective focal points

    Parameters
    ----------
    FF - a list of Near2FarFields objects, representing a 3d matrix with:
        [0] the points where the fields are calculated
        [1] the different frequencies at which they are calculated
        [2] the values for Ex, Ey, Ez, Hx, Hy, Hz

    Returns
    -------
    The value of the objective function to maximize
    """
    return min(np.abs(FF[0, 0, 2] ** 2),   # focus red left
               np.abs(FF[1, 1, 2] ** 2))  # and blue right, and third field (Ez)


# Optimization object
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J1],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    maximum_run_time=2000,
)

plt.figure()
opt.plot2D(True)
plt.savefig("./" + scriptName + "/optimizationDesign.png")

# Gradient
evaluation_history = []  # Keep track of objective function evaluations
cur_iter = [0]  # Iteration


def f(v, gradient, cur_beta):
    if cur_iter[0] != 0:
        estimatedSimulationTime = (datetime.datetime.now() - start) * totalIterations / cur_iter[0]
        print("Current iteration: {}".format(cur_iter[0]) + "; " + str(100 * cur_iter[0] / totalIterations) +
              "% completed ; eta at " + str(start + estimatedSimulationTime))
    else:
        print("Current iteration: {}".format(cur_iter[0]))
    cur_iter[0] += 1

    f0, dJ_du = opt([mapping(v, eta_i, cur_beta)])  # compute objective and gradient

    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, cur_beta, np.sum(dJ_du, axis=1)
        )  # backprop

    evaluation_history.append(np.real(f0))  # add objective function evaluation to list

    # plt.figure() # Plot current design
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
    plt.savefig("./" + scriptName + "/img_" + str(cur_iter[0]) + ".png")

    return np.real(f0)


# Method of moving  asymptotes
algorithm = nlopt.LD_MMA
n = Nx * Ny  # number of parameters

# Initial guess
file_path = "x.npy"
with open(file_path, 'rb') as file:
    x = np.load(file)

# lower and upper bounds
lb = np.zeros((Nx * Ny,))
ub = np.ones((Nx * Ny,))

cur_beta = 4
beta_scale = 2
num_betas = 6
update_factor = 12
totalIterations = num_betas * update_factor
ftol = 1e-3
start = datetime.datetime.now()
# for iters in range(num_betas):
#     solver = nlopt.opt(algorithm, n)
#     solver.set_lower_bounds(lb)
#     solver.set_upper_bounds(ub)
#     solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
#     solver.set_maxeval(update_factor)  # stop when 12 iterations are reached
#     solver.set_ftol_rel(ftol)  # or when we converged
#     x[:] = solver.optimize(x)
#     cur_beta = cur_beta * beta_scale
#     estimatedSimulationTime = (datetime.datetime.now() - start) * num_betas / (iters + 1)
#     print("Current iteration: {}".format(cur_iter[0]) + "; " + str(100 * cur_iter[0] / totalIterations) +
#           "% completed ; eta at " + str(start + estimatedSimulationTime))
#     sendNotification("Opitimization " + str(100 * (iters + 1) / num_betas) + " % completed; eta at " +
#                         str(start + estimatedSimulationTime))
#     np.save("./" + scriptName + "/x", x)


cur_beta = 256
# Plot final design
opt.update_design([mapping(x, eta_i, cur_beta)])
# plt.figure()
# ax = plt.gca()
# opt.plot2D(
#     False,
#     ax=ax,
#     plot_sources_flag=False,
#     plot_monitors_flag=False,
#     plot_boundaries_flag=False,
# )
# circ = Circle((2, 2), minimum_length / 2)
# ax.add_patch(circ)
# ax.axis("off")
# plt.savefig("./" + scriptName + "/finalDesign.png")
# np.save("./" + scriptName + "/x", x)
# # Check intensities in optimal design
# f0, dJ_du = opt([mapping(x, eta_i, cur_beta // 2)], need_gradient=False)
# frequencies = opt.frequencies
#
# try:
#     intensities = [np.abs(opt.get_objective_arguments()[0][0, 0, 2]) ** 2,
#                    np.abs(opt.get_objective_arguments()[0][1, 1, 2]) ** 2]
#
#     # Plot intensities
#     plt.figure()
#     plt.plot([1 / freq for freq in frequencies], intensities, "-o")
#     plt.grid(True)
#     plt.xlabel("Wavelength (microns)")
#     plt.ylabel("|Ez|^2 Intensities")
#     plt.savefig("./" + scriptName + "/intensities.png")
# except:
#     print('exception in calculating intensities')

# Plot evaluation history
# plt.figure()
# plt.plot([i for i in range(len(evaluation_history))], evaluation_history, "-o")
# plt.grid(True)
# plt.xlabel("Iteration")
# plt.ylabel("Minimum field")
# plt.savefig("./" + scriptName + "/objective.png")
#
# plt.figure()
# opt.plot2D(fields=mp.Ez,
#            field_parameters={'interpolation': 'spline36', 'cmap': 'RdBu'})
# plt.grid(True)
# plt.xlabel("µm")
# plt.ylabel("µm")
#
# plt.savefig("./" + scriptName + "/fields.png")
# plt.show()

for freq in frequencies:
    opt.sim = mp.Simulation(
        cell_size=mp.Vector3(Sx, 70),
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        default_material=Air,
        resolution=resolution,
    )
    src = mp.ContinuousSource(frequency=freq, fwidth=width*freq)
    source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
    opt.sim.change_sources(source)

    opt.sim.run(until=1000)
    plt.figure(figsize=(10, 20))
    opt.sim.plot2D(fields=mp.Ez)
    fileName = f"./" + scriptName + "/fieldAtWavelength" + str(1/freq) + "-extra-long.png"
    plt.savefig(fileName)

plt.close()

