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


# from scipy import special, signal
import datetime
import requests  # send notifications

scriptName = "metalens_1layer_3d_2"
#mp.divide_parallel_processes(16)


def sendNotification(message):
    token = "5421873058:AAFIKUk8fSksmo2qe9rHZ0dmYo0CI12fYyU"
    myuserid = 6297309186
    method = '/sendMessage'
    url = f"https://api.telegram.org/bot{token}"
    params = {"chat_id": myuserid, "text": message}
    try:
        r = requests.get(url + method, params=params)
    except:
        print("No internet connection: couldn't send notification...")

def sendPhoto(image_path):
    token = "5421873058:AAFIKUk8fSksmo2qe9rHZ0dmYo0CI12fYyU"
    myuserid = 6297309186
    data = {"chat_id": myuserid}
    url = f"https://api.telegram.org/bot{token}" + "/sendPhoto"
    with open(image_path, "rb") as image_file:
        try:
            ret = requests.post(url, data=data, files={"photo": image_file})
            return ret.json()
        except:
            print("No internet connection: couldn't send notification...")


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

mp.verbosity(1)  # amount of info printed during simulation

# Materials
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.45)
NbOx = mp.Medium(index=2.5)
TiOx = mp.Medium(index=2.7) # 550 nm / 2.7 = 204 nm --> 20.4 nm resolution = 49
Air = mp.Medium(index=1.0)

# Dimensions
design_region_width = 10
design_region_height = 0.24

# Boundary conditions
pml_size = 1

resolution = 60

# System size
Sx = 2 * pml_size + design_region_width
Sz = 2 * pml_size + design_region_height + 1.5
cell_size = mp.Vector3(Sx, Sx, Sz)

# Frequencies
nf = 1  # Amount of frequencies studied
# frequencies = np.array([1 / 1.5, 1 / 1.55, 1 / 1.6])
frequencies = [0.6] # 1. / np.linspace(0.55, 0.65, 3)

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
pml_layers = [mp.Absorber(pml_size)]

fwidth = 0.4  # Relative width of frequency
source_center = [0, 0, -(design_region_height / 2 + 0.5)]  # Source 0.4 um below lens
source_size = mp.Vector3(design_region_width, design_region_width, 0)  # Source covers width of lens
srcs = [mp.GaussianSource(frequency=fcen, fwidth=fwidth) for fcen in frequencies] # Gaussian source
source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center) for src in srcs]

# Amount of variables
Nx = int(design_region_resolution * design_region_width)
Ny = int(design_region_resolution * design_region_width)
Nz = 1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, TiOx, grid_type="U_MEAN")
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
    filtered_field = mpa.conic_filter(  # remain minimum feature size
        x,
        filter_radius,
        design_region_width,
        design_region_width,
        design_region_resolution,
    )

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)  # Make binary

    # interpolate to actual materials
    return projected_field.flatten()


# Lens geometry
geometry = [
	mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
]

geometry.append(mp.Block(
        center = mp.Vector3(z=-(half_total_height + Sy/2) / 2),
        size=mp.Vector3(x = Sx, y = Sx, z = (Sy/2 - half_total_height)),
        material=SiO2
    ))

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
    eps_averaging=False,
)

# Focus point, 6 uM beyond centre of lens
far_x = [mp.Vector3(0, 0, 6)]
NearRegions = [  # region from which fields at focus point will be calculated
    mp.Near2FarRegion(
        center=mp.Vector3(0, 0, design_region_height / 2 + 0.3),  # 0.3 um above lens
        size=mp.Vector3(design_region_width, design_region_width, 0),  # spans design region
        weight=+1,  # field contribution is positive (real)
    )
]

FarFields = mpa.Near2FarFields(sim, NearRegions, far_x)  # Far-field object
ob_list = [FarFields]


def J1(FF):
    print(FF)
    return npa.mean(npa.abs(FF[0, :, 0]) ** 2)  # only first (only point), mean of all frequencies, and third field (Ez)


# Optimization object
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J1],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    maximum_run_time=500,
)

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
    print("gradient")
    print(gradient)
    print("f0")
    print(f0)
    print("dJ_du")
    print(dJ_du)
    print("dJ_du size")
    print(dJ_du.size)
    print("v")
    print(v)
    print("v.size")
    print(v.size)
    if gradient.size > 0:
        qsdf = tensor_jacobian_product(mapping, 0)(v, eta_i, cur_beta, np.sum(dJ_du, axis=1))  # backprop
        print(qsdf)
        gradient[:] = qsdf

    evaluation_history.append(f0)  # add objective function evaluation to list

    np.save("./" + scriptName + "/x_" + str(cur_iter[0]), x)

    opt.plot2D(output_plane=output_plane)
    plt.savefig(fname=scriptName + "/" + str(cur_iter[0]) + ".png")

    return f0

# Method of moving  asymptotes
algorithm = nlopt.LD_MMA
n = Nx * Ny  # number of parameters

# Initial guess
x = np.random.rand(n)
opt.update_design([mapping(x, eta_i, 4)])

# lower and upper bounds
lb = np.zeros(n)
ub = np.ones(n)


print("plot")
output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(15, 0, 10))
plt.figure()
opt.plot2D(output_plane=output_plane, plot_monitors_flag=True)
plt.savefig(scriptName + "/design_XZ.png")

output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(9, 9, 0))
plt.figure()
opt.plot2D(output_plane=output_plane)
plt.savefig(scriptName + "/design_layer1.png")

cur_beta = 8 #4
beta_scale = 2
num_betas = 5 #6
update_factor = 10 #20
totalIterations = num_betas * update_factor
ftol = 1e-2
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
    np.save("./" + scriptName + "/x_" + str(cur_iter[0]), x)

# Check intensities in optimal design
f0, dJ_du = opt([mapping(x, eta_i, cur_beta // 2)], need_gradient=False)
frequencies = opt.frequencies

intensities = np.abs(opt.get_objective_arguments()[0][0, :, 0])

# Plot intensities
plt.figure()
plt.plot(1 / frequencies, intensities, "-o")
plt.grid(True)
plt.xlabel("Wavelength (microns)")
plt.ylabel("|Ez|^2 Intensities")
plt.savefig("./" + scriptName + "/intensities.png")

np.save("./" + scriptName + "/x", x)

# plot evaluation history	
plt.figure()
plt.plot([i for i in range(len(evaluation_history))], evaluation_history, "-o")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Minimum field")
plt.savefig("./" + scriptName + "/objective.png")

# Plot final design
opt.update_design([mapping(x, eta_i, 256)])
opt.plot2D(output_plane=output_plane)
plt.savefig("./" + scriptName + "/finalDesign.png")

# sendNotification("Simulation finished")
