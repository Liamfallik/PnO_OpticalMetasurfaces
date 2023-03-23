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
import mlab

from scipy import special, signal
import datetime
import requests  # send notifications

scriptName = "metalens_1layer_3d_img"


def sendNotification(message):
    token = "5421873058:AAFIKUk8fSksmo2qe9rHZ0dmYo0CI12fYyU"
    myuserid = 6297309186
    method = '/sendMessage'
    url = f"https://api.telegram.org/bot{token}"
    params = {"chat_id": myuserid, "text": message}
    r = requests.get(url + method, params=params)


def sendPhoto(image_path):
    token = "5421873058:AAFIKUk8fSksmo2qe9rHZ0dmYo0CI12fYyU"
    myuserid = 6297309186
    data = {"chat_id": myuserid}
    url = f"https://api.telegram.org/bot{token}" + "/sendPhoto"
    with open(image_path, "rb") as image_file:
        ret = requests.post(url, data=data, files={"photo": image_file})
    return ret.json()


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


# checking if the directory demo_folder
# exist or not.
if not os.path.exists("./" + scriptName):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs("./" + scriptName)

mp.verbosity(0)  # amount of info printed during simulation

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
Sz = 2 * pml_size + design_region_height + 5
cell_size = mp.Vector3(Sx, Sx, Sz)

# Frequencies
nf = 3  # Amount of frequencies studied
# frequencies = np.array([1 / 1.5, 1 / 1.55, 1 / 1.6])
frequencies = 1. / np.linspace(1.5, 1.6, 3)

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

fcen = 1 / 1.55  # Middle frequency of source
width = 0.2  # Relative width of frequency
fwidth = width * fcen  # Absolute width of frequency
source_center = [0, 0, -(design_region_height / 2 + 1.5)]  # Source 1.5 µm below lens
source_size = mp.Vector3(design_region_width, design_region_width, 0)  # Source covers width of lens
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)  # Gaussian source
source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]

# Amount of variables
Nx = int(design_region_resolution * design_region_width)
Ny = int(design_region_resolution * design_region_width)
Nz = 1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), Air, Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(design_region_width, design_region_width, design_region_height),
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

    projected_field = (
                              npa.flipud(projected_field) + projected_field
                      ) / 2  # left-right symmetry

    # interpolate to actual materials
    return projected_field.flatten()


# Lens geometry
geometry = [
    mp.Cylinder(
        center=design_region.center, radius=design_region_width/2, material=design_variables, height=design_region_height
    ),
    # mp.Block(center=design_region.center, size=design_region.size, material=design_variables, e1=mp.Vector3(x=-1))
    #
    # The commented lines above impose symmetry by overlapping design region with the same design variable. However,
    # currently there is an issue of doing that; instead, we use an alternative approach to impose symmetry.
    # See https://github.com/NanoComp/meep/issues/1984 and https://github.com/NanoComp/meep/issues/2093
]
geometry_plot = [mp.Cylinder(
        center=design_region.center, radius=design_region_width/2, material=Si, height=design_region_height
    ),
]

# Set-up simulation object
kpoint = mp.Vector3()
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air,
    symmetries=[mp.Mirror(direction=mp.X), mp.Mirror(direction=mp.Y)],
    resolution=resolution,
)

# Focus point, 15 µm beyond centre of lens
far_x = [mp.Vector3(0, 0, 15)]
NearRegions = [  # region from which fields at focus point will be calculated
    mp.Near2FarRegion(
        center=mp.Vector3(0, 0, design_region_height / 2 + 1.5),  # 1.5 µm above lens
        size=mp.Vector3(design_region_width, design_region_width, 0),  # spans design region
        weight=+1,  # field contribution is positive (real)
    )
]
FarFields = mpa.Near2FarFields(sim, NearRegions, far_x)  # Far-field object
ob_list = [FarFields]


def J1(FF):
    return npa.mean(npa.abs(FF[0, :, 2]) ** 2)  # only first (only point), mean of all frequencies, and third field (Ez)


# Optimization object
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J1],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    maximum_run_time=2000,
)

print("plot")
# sim.plot3D(save_to_image=True, image_name="{0}.png".format(scriptName))
print("plot2")
# sim.plot3D()
output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(design_region_width, design_region_width, 0))
plt.figure()
sim.plot2D(output_plane=output_plane)
plt.savefig(scriptName + ".png")

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

    sim.plot2D(output_plane=output_plane)
    plt.savefig(fname=cur_iter[0] + ".png")

    return np.real(f0)

# Method of moving  asymptotes
algorithm = nlopt.LD_MMA
n = Nx * Ny * Nz  # number of parameters

# Initial guess
x = np.ones((n,)) * 0.5  # average everywhere

# lower and upper bounds
lb = np.zeros((Nx * Ny * Nz,))
ub = np.ones((Nx * Ny * Nz,))

cur_beta = 4
beta_scale = 2
num_betas = 6
update_factor = 12
totalIterations = num_betas * update_factor
ftol = 1e-5
start = datetime.datetime.now()
print("Opitimization started at " + str(start))
# sendNotification("Opitimization started at " + str(start))
for iters in range(num_betas):
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
    solver.set_maxeval(update_factor)  # stop when 12 iterations or reached
    solver.set_ftol_rel(ftol)  # or when we converged
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale

    estimatedSimulationTime = (datetime.datetime.now() - start) * num_betas / (iters + 1)
    print("Current iteration: {}".format(cur_iter[0]) + "; " + str(100 * cur_iter[0] / totalIterations) +
          "% completed ; eta at " + str(start + estimatedSimulationTime))
    # sendNotification("Opitimization " + str(100 * (iters + 1) / num_betas) + " % completed; eta at " +
    #                  str(start + estimatedSimulationTime))
    np.save("./" + scriptName + "/x", x)

# Plot final design
sim.plot2D(output_plane=output_plane)
plt.savefig("./" + scriptName + "/finalDesign.png")

# Check intensities in optimal design
f0, dJ_du = opt([mapping(x, eta_i, cur_beta // 2)], need_gradient=False)
frequencies = opt.frequencies

intensities = np.abs(opt.get_objective_arguments()[0][0, :, 2]) ** 2

# Plot intensities
plt.figure()
plt.plot(1 / frequencies, intensities, "-o")
plt.grid(True)
plt.xlabel("Wavelength (microns)")
plt.ylabel("|Ez|^2 Intensities")
plt.savefig("./" + scriptName + "/intensities.png")

np.save("./" + scriptName + "/x", x)

# sendNotification("Simulation finished")

