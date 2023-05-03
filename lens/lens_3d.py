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

scriptName = "metalens_2layers_3d"
num_layers = 2
start_from_direct = True
symmetry = True # NOT IMPLEMENTED YET
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


def symmetrize(x, Nx, Ny):
    x2 = np.reshape(x, [Nx, Ny])
    for i in range(Nx // 2):
        for j in range(Ny // 2):
            x2[i, j] = (x2[i, j] + x2[-(i+1), j] + x2[i, -(j+1)] + x2[-(i+1), -(j+1)]) / 4
            x2[-(i+1), j] = x2[i, j]
            x2[i, -(j+1)] = x2[i, j]
            x2[-(i+1), -(j+1)] = x2[i, j]

    return np.reshape(x2, [Nx * Ny])


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
design_region_height = [0.48 / (num_layers + 1)]*num_layers
spacing = 0
half_total_height = sum(design_region_height) / 2 + (num_layers - 1) * spacing / 2

# Boundary conditions
pml_size = 0.5
resolution = 50

# System size
Sx = 2 * pml_size + design_region_width
Sz = 2 * pml_size + half_total_height * 2 + 1
cell_size = mp.Vector3(Sx, Sx, Sz)

# Frequencies
nf = 1  # Amount of frequencies studied
# frequencies = np.array([1 / 1.5, 1 / 1.55, 1 / 1.6])
frequencies = np.array([0.6]) # 1. / np.linspace(0.55, 0.65, 3)

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
source_pos = -(half_total_height + 0.3)
source_center = [0, 0, source_pos] # Source 0.3 µm below lens
source_size = mp.Vector3(design_region_width, design_region_width, 0)  # Source covers width of lens
srcs = [mp.GaussianSource(frequency=fcen, fwidth=fwidth) for fcen in frequencies] # Gaussian source
source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center) for src in srcs]

# Amount of variables
Nx = int(design_region_resolution * design_region_width)
Ny = int(design_region_resolution * design_region_width)
Nz = 1

design_variables = [mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, TiOx, grid_type="U_MEAN") for i in range(num_layers)]
design_regions = [mpa.DesignRegion(
    design_variables[i],
    volume=mp.Volume(
        center=mp.Vector3(z=-half_total_height + 0.5 * design_region_height[i] + sum(design_region_height[:i]) + i * spacing),
        size=mp.Vector3(design_region_width, design_region_width, design_region_height[i]),
    ),
) for i in range(num_layers)]
# design_region = mpa.DesignRegion(
#     design_variables,
#     volume=mp.Volume(
#         center=mp.Vector3(),
#         size=mp.Vector3(design_region_width, design_region_width, design_region_height),
#     ),
# )


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

    # symmetry
    if symmetry:
        projected_field = symmetrize(projected_field, Nx, Ny)

    # interpolate to actual materials
    return projected_field.flatten()


# Lens geometry
geometry = [
	mp.Block(
        center=design_region.center, size=design_region.size, material=design_region.design_parameters
    )
for design_region in design_regions]

geometry.append(mp.Block(
        center = mp.Vector3(z=-(half_total_height + Sz/2) / 2),
        size=mp.Vector3(x = Sx, y = Sx, z = (Sz/2 - half_total_height)),
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
    symmetries=[mp.Mirror(direction=mp.X), mp.Mirror(direction=mp.Y)] if symmetry else Nones,
    resolution=resolution,
    eps_averaging=False,
)

# Focus point, 6 uM beyond centre of lens
focal_point = 6
far_x = [mp.Vector3(0, 0, focal_point)]
NearRegions = [  # region from which fields at focus point will be calculated
    mp.Near2FarRegion(
        center=mp.Vector3(0, 0, half_total_height + 0.3),  # 0.3 um above lens
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
    design_regions=design_regions,
    frequencies=frequencies,
    maximum_run_time=500, # 500
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

    if gradient.size > 0:
        qsdf = tensor_jacobian_product(mapping, 0)(v, eta_i, cur_beta, np.sum(dJ_du, axis=1))  # backprop
        # print(qsdf)
        gradient[:] = qsdf

    evaluation_history.append(f0)  # add objective function evaluation to list

    np.save("./" + scriptName + "/x_" + str(cur_iter[0]), x)

    output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(design_region_width, design_region_width, 0))
    plt.figure() # Plot current design
    # ax = plt.gca()

    opt.update_design([mapping(v, eta_i, cur_beta)])
    opt.plot2D(
        False,
        # ax=ax,
        plot_sources_flag=False,
        plot_monitors_flag=False,
        plot_boundaries_flag=False,
        output_plane=output_plane
    )
    circ = Circle((2, 2), minimum_length / 2)
    ax.add_patch(circ)
    ax.axis("off")
    plt.savefig("./" + scriptName + "/" + "/img_{" + str(cur_iter[0]) + "}.png")

    plt.close()

    return f0

# Method of moving  asymptotes
algorithm = nlopt.LD_MMA
n = Nx * Ny * num_layers # number of parameters

seed = 240 # make sure starting conditions are random, but always the same. Change seed to change starting conditions
np.random.seed(seed)
# Initial guess
reshaped_x = np.zeros([num_layers, Nx, Nx])
if start_from_direct:
    random_perturbation = 0.5 # Something wrong with randomization
    phase0 = random.random()# (1-0.9*0.5**num_layers)
    for k in range(Ny//2, Ny):
        for j in range(Nx//2, Nx):
            phase = (phase0 - np.sqrt(focal_point**2 + ((j - Nx//2) / design_region_resolution)**2 +
                                      ((k - Ny//2) / design_region_resolution)**2) * frequencies[nf // 2]) % 1
            phase_step = int(phase * 2**num_layers)
            for i in range(num_layers):
                if phase_step % (2**(num_layers - i)) // 2**(num_layers - i - 1) != 0:
                    reshaped_x[i, j, k] = 3.0 if (symmetry and (j != Nx // 2 and k != Ny // 2)) else 0.75
                else:
                    reshaped_x[i, j, k] = 1.0 if symmetry else 0.25

    x = np.reshape(reshaped_x, [n]) + random_perturbation * (-0.5 + np.random.rand(n))
    if symmetry:
        x = symmetrize(x, Nx, Ny)
else:
    x = np.random.rand(n)

start_beta = 4 # 4
opt.update_design([mapping(x, eta_i, start_beta)])

# lower and upper bounds
lb = np.zeros(n)
ub = np.ones(n)

print("plot")
output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(Sx, 0, 10))
plt.figure()
opt.plot2D(output_plane=output_plane, plot_monitors_flag=True)
plt.savefig(scriptName + "/design_XZ.png")

output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(design_region_width, design_region_width, 0))
plt.figure()
opt.plot2D(output_plane=output_plane)
plt.savefig(scriptName + "/design_layer1.png")

cur_beta = start_beta
beta_scale = 4
num_betas = 3 #6
update_factor = 10 #20
totalIterations = num_betas * update_factor
ftol = 1e-2
start = datetime.datetime.now()


with open("./" + scriptName + "/used_variables.txt", 'w') as var_file:
    var_file.write("num_layers \t" + str(num_layers) + "\n")
    var_file.write("symmetry \t" + str(symmetry) + "\n")
    var_file.write("design_region_width \t" + str(design_region_width) + "\n")
    var_file.write("design_region_height \t" + str(design_region_height) + "\n")
    var_file.write("design_region_resolution \t" + str(design_region_resolution) + "\n")
    var_file.write("minimum_length \t" + str(minimum_length) + "\n")
    var_file.write("spacing \t" + str(spacing) + "\n")
    # var_file.write("empty_space \t" + str(empty_space) + "\n")
    var_file.write("pml_size \t" + str(pml_size) + "\n")
    var_file.write("resolution \t" + str(resolution) + "\n")
    var_file.write("wavelengths \t" + str(1/frequencies) + "\n")
    var_file.write("fwidth \t" + str(fwidth) + "\n")
    var_file.write("focal_point \t" + str(focal_point) + "\n")
    var_file.write("start_beta \t" + str(start_beta) + "\n")
    var_file.write("beta_scale \t" + str(beta_scale) + "\n")
    var_file.write("num_betas \t" + str(num_betas) + "\n")
    var_file.write("update_factor \t" + str(update_factor) + "\n")
    var_file.write("ftol \t" + str(ftol) + "\n")
    var_file.write("seed \t%d" % seed + "\n")

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
    sendNotification("Opitimization " + str(100 * (iters + 1) / num_betas) + " % completed; eta at " +
                     str(start + estimatedSimulationTime))
    np.save("./" + scriptName + "/x_" + str(cur_iter[0]), x)

cur_beta = cur_beta / beta_scale

# Check intensities in optimal design
f0, dJ_du = opt([mapping(x, eta_i, cur_beta)], need_gradient=False)
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
plt.ylabel("Relative intensity at focal point")
plt.savefig("./" + scriptName + "/objective.png")

# Plot final design
opt.update_design([mapping(x, eta_i, cur_beta)])
opt.plot2D(output_plane=output_plane)
plt.savefig("./" + scriptName + "/finalDesign.png")

# sendNotification("Simulation finished")
