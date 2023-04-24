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
scriptName = "metalens_img_StochOpt_test22"
symmetry = True # Impose symmetry around x = 0 line

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

mp.verbosity(0) # amount of info printed during simulation

# Materials
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.45)
NbOx = mp.Medium(index=2.5)
TiOx = mp.Medium(index=2.7) # 550 nm / 2.7 = 204 nm --> 20.4 nm resolution = 49
Air = mp.Medium(index=1.0)

# Dimensions
num_layers = 6 # amount of layers
design_region_width = 10 # width of layer
design_region_height = [0.0686]*num_layers # height of layer
spacing = 0 # spacing between layers
half_total_height = sum(design_region_height) / 2 + (num_layers - 1) * spacing / 2
empty_space = 0 # free space in simulation left and right of layer

# Boundary conditions
pml_size = 1.0 # thickness of absorbing boundary layer

resolution = 50 # 50 --> amount of grid points per µm; needs to be > 49 for TiOx and 0.55 µm

# System size
Sx = 2 * pml_size + design_region_width + 2 * empty_space
Sy = 2 * pml_size + half_total_height * 2 + 2
cell_size = mp.Vector3(Sx, Sy)

# Frequencies
nf = 3 # Amount of frequencies studied
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
pml_layers = [mp.PML(pml_size)]

# Source
fcen = frequencies[1]
fwidth = 0.03 # 0.2
factor = 2/(fwidth * np.sqrt(2*pi))
source_pos = -(half_total_height + 0.4)
source_center = [0, source_pos, 0] # Source 0.4 µm below lens
source_size = mp.Vector3(design_region_width + 2*empty_space, 0, 0) # Source covers width of lens
# src = mp.GaussianSource(frequency=fcen, fwidth=fwidth) # Gaussian source
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
srcs = [mp.GaussianSource(frequency=fcen, fwidth=fwidth) for fcen in frequencies] # Gaussian source
source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center) for src in srcs]

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

    if symmetry:
        filtered_field = (
            npa.flipud(filtered_field) + filtered_field
        ) / 2  # left-right symmetry

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)  # Make binary

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

# Set-up simulation object
kpoint = mp.Vector3()

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air, # Air
    symmetries=[mp.Mirror(direction=mp.X)] if symmetry else None,
    resolution=resolution,
)

# Focus point, 7.5 µm beyond centre of lens
focal_point = 6
far_x = [mp.Vector3(0, focal_point, 0)]
NearRegions = [ # region from which fields at focus point will be calculated
    mp.Near2FarRegion(
        center=mp.Vector3(0, half_total_height + 0.4), # 0.4 µm above lens
        size=mp.Vector3(design_region_width + 2*empty_space, 0), # spans design region
        weight=+1, # field contribution is positive (real)
    )
]
FarFields = mpa.Near2FarFields(sim, NearRegions, far_x) # Far-field object
ob_list = [FarFields]


def J1(FF):
    print(FF[0, :, 2])
    return npa.mean(npa.abs(FF[0, :, 2]) ** 2) # only first (only point), mean of all frequencies, and third field (Ez)

# Optimization object
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J1],
    objective_arguments=ob_list,
    design_regions=design_regions,
    frequencies=frequencies,
    maximum_run_time=2000,
)

plt.figure()
opt.plot2D(True)
plt.savefig("./" + scriptName + "/optimizationDesign.png")

# Gradient
evaluation_history = [] # Keep track of objective function evaluations
cur_iter = [0] # Iteration


def f(v, gradient, cur_beta):
    if cur_iter[0] != 0:
        estimatedSimulationTime = (datetime.datetime.now() - start) * totalIterations / cur_iter[0]
        print("Sample: {}".format(sample_nr) + "; " + "Current iteration: {}".format(cur_iter[0]) + "; " + str(100 * cur_iter[0] / totalIterations) +
              "% completed ; eta at " + str(start + estimatedSimulationTime))
    else:
        print("Current iteration: {}".format(cur_iter[0]))
    cur_iter[0] += 1
    reshaped_v = np.reshape(v, [num_layers, Nx])

    f0, dJ_du = opt([mapping(reshaped_v[i, :], eta_i, cur_beta) for i in range(num_layers)]) # compute objective and gradient
    # shape of dJ_du [# degrees of freedom, # frequencies] or [# design regions, # degrees of freedom, # frequencies]


    if gradient.size > 0:
        if isinstance(dJ_du[0][0], list) or isinstance(dJ_du[0][0], np.ndarray):
            gradi = [tensor_jacobian_product(mapping, 0)(
                reshaped_v[i, :], eta_i, cur_beta, np.sum(dJ_du[i], axis=1)
            ) for i in range(num_layers)] # backprop
        else:
            gradi = tensor_jacobian_product(mapping, 0)(
                reshaped_v, eta_i, cur_beta, np.sum(dJ_du, axis=1)) # backprop

        gradient[:] = np.reshape(gradi, [n])

    evaluation_history.append(np.real(f0)) # add objective function evaluation to list
    print(evaluation_history[-1])

    plt.figure() # Plot current design
    ax = plt.gca()
    opt.update_design([mapping(reshaped_v[i, :], eta_i, cur_beta) for i in range(num_layers)])
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
    plt.savefig("./" + scriptName + "/" + scriptName_i + "/img_{" + str(cur_iter[0]) + ".png")

    # Efield = opt.sim.get_epsilon()
    # plt.figure()
    # plt.plot(Efield ** 2)

    plt.figure()
    # plt.fill_between(np.arange(num_samples), ubev, lbev, alpha=0.3)
    plt.plot(evaluation_history, "o-")
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("FOM")
    fileName = f"./" + scriptName + "/" + scriptName_i + "/FOM.png"
    plt.savefig(fileName)
    plt.close()



    return np.real(f0)


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

num_samples = 1#5
# store best objective value
best_f0 = 0
best_design = None
best_nr = None
f0s = np.zeros([num_samples, nf])

start0 = datetime.datetime.now()
for sample_nr in range(num_samples):
    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[J1],
        objective_arguments=ob_list,
        design_regions=design_regions,
        frequencies=frequencies,
        maximum_run_time=2000,
    )

    # Create the animation
    animate = Animate2D(
        fields=None,
        # realtime=True,
        eps_parameters={'contour': False, 'alpha': 1, 'frequency': frequencies[1]},
        plot_sources_flag=False,
        plot_monitors_flag=False,
        plot_boundaries_flag=False,
        update_epsilon=True,  # required for the geometry to update dynamically
        nb=False  # True required if running in a Jupyter notebook
    )
    animateField = Animate2D(
        fields=mp.Ez,
        # realtime=True,
        eps_parameters={'contour': False, 'alpha': 1, 'frequency': frequencies[1]},
        plot_sources_flag=True,
        plot_monitors_flag=True,
        plot_boundaries_flag=True,
        update_epsilon=True,  # required for the geometry to update dynamically
        nb=False  # True required if running in a Jupyter notebook
    )
    # This will trigger the animation at the end of each simulation
    opt.step_funcs = [mp.at_end(animate), mp.at_end(animateField)]

    # Method of moving  asymptotes
    algorithm = nlopt.LD_MMA  # nlopt.LD_MMA
    n = Nx * num_layers  # number of parameters

    # lower and upper bounds
    lb = np.zeros((n,))
    ub = np.ones((n,))

    x = np.random.rand(n) * 0.6
    # reshaped_x = np.zeros([num_layers, Nx])

    # phase0 = (focal_point * frequencies[1]) % 1 + random.random()# (1-0.9*0.5**num_layers)
    # for j in range(Nx//2, Nx):
    #     phase = (phase0 - np.sqrt(focal_point**2 + ((j - Nx//2) / design_region_resolution)**2) * frequencies[1]) % 1
    #     phase_step = int(phase * 2**num_layers)
    #     for i in range(num_layers):
    #         if phase_step % (2**(num_layers - i)) // 2**(num_layers - i - 1) != 0:
    #             reshaped_x[i, j] = 1.5 if symmetry else 0.75
    #         else:
    #             reshaped_x[i, j] = 0.5 if symmetry else 0.25


    # x = np.reshape(reshaped_x, [n])# + 0.5 * (-0.5 + np.random.rand(n))
    file_path = "x.npy"
    with open(file_path, 'rb') as file:
        x = np.load(file)

    # if symmetry:
    #     for i in range(num_layers):
    #         x[Nx*i:Nx*(i+1)] = (npa.flipud(x[Nx*i:Nx*(i+1)]) + x[Nx*i:Nx*(i+1)]) / 2  # left-right symmetry
    # x[Nx:] = np.zeros(n - Nx)

    scriptName_i = "sample_" + str(sample_nr)
    # checking if the directory demo_folder
    # exist or not.
    if not os.path.exists("./" + scriptName + "/" + scriptName_i):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs("./" + scriptName + "/" + scriptName_i)

    evaluation_history = []  # Keep track of objective function evaluations
    cur_iter = [0]  # Iteration

    # Plot first design
    reshaped_x = np.reshape(x, [num_layers, Nx])
    mapped_x = [mapping(reshaped_x[i, :], eta_i, 4) for i in range(num_layers)]
    opt.update_design(mapped_x)
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
    plt.savefig("./" + scriptName + "/" + scriptName_i + "/firstDesign.png")

    # Optimization
    cur_beta = 4*2**6 # 4
    beta_scale = 2 # 2
    num_betas = 7*0 # 6
    local_opt_steps = 15 # 12
    monte_carlo_steps = 5
    random_perturbation = 0.5
    perturbation_scale = 1.2
    totalIterations = num_betas * local_opt_steps * monte_carlo_steps
    ftol = 1e-4 # 1e-5
    start = datetime.datetime.now()
    print("Opitimization started at " + str(start))
    sendNotification("Opitimization started at " + str(start))
    best_x = x
    best_f = 1

    for iters in range(num_betas):
        if iters == num_betas - 1:
            local_opt_steps = 2*local_opt_steps
        best_f = 0
        for monte_step in range(monte_carlo_steps):
            solver = nlopt.opt(algorithm, n)
            solver.set_lower_bounds(lb)
            solver.set_upper_bounds(ub)
            solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
            solver.set_maxeval(local_opt_steps) # stop when 12 iterations are reached
            solver.set_ftol_rel(ftol)  # or when we converged
            if monte_step != 0:
                x = np.clip(best_x + random_perturbation * (1 - 2*np.random.rand(n)), 0, 1)
            else:
                x = best_x

            x = solver.optimize(x)

            if evaluation_history[-1] > best_f:
                print(evaluation_history[-1])
                print(best_f)
                best_f = evaluation_history[-1]
                best_x = x

            estimatedSimulationTime = (datetime.datetime.now() - start) * (num_betas / (iters + (monte_step + 1) / monte_carlo_steps))
            print("Current iteration: {}".format(cur_iter[0]) + "; " + str(100 * cur_iter[0] / totalIterations) +
                  "% completed ; eta at " + str(start + estimatedSimulationTime))

        cur_beta = cur_beta * beta_scale
        random_perturbation = random_perturbation / perturbation_scale

        np.save("./" + scriptName + "/" + scriptName_i + "/x", best_x)
        plt.close()

    cur_beta = cur_beta / beta_scale
    x = best_x
    # Plot final design
    reshaped_x = np.reshape(x, [num_layers, Nx])
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
    plt.savefig("./" + scriptName + "/" + scriptName_i + "/finalDesign.png")

    # Check intensities in optimal design
    f0, dJ_du = opt([mapping(reshaped_x[i, :], eta_i, cur_beta) for i in range(num_layers)], need_gradient=False)
    frequencies = opt.frequencies

    if best_f > best_f0:
        best_f0 = best_f
        best_design = best_x
        best_nr = sample_nr

    print("Objective_value = " + str(best_f))

    intensities = np.abs(opt.get_objective_arguments()[0][0, :, 2]) ** 2
    print(opt.get_objective_arguments())

    f0s[sample_nr, :] = intensities

    # Plot intensities
    plt.figure()
    plt.plot(1 / frequencies, intensities, "-o")
    plt.grid(True)
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("|Ez|^2 Intensities")
    plt.savefig("./" + scriptName + "/" + scriptName_i + "/intensities.png")

    np.save("./" + scriptName + "/" + scriptName_i + "/v", x)

    animate.to_gif(fps=5, filename="./" + scriptName + "/" + scriptName_i + "/animation.gif")

    Sy2 = 20
    geometry.append(mp.Block(
        center=mp.Vector3(y=-(Sy2 / 2 + Sy / 2) / 2),
        size=mp.Vector3(x=Sx, y=(Sy2 / 2 - Sy / 2)),
        material=SiO2
    ))

    # Plot fields
    for freq in frequencies:
        opt.sim = mp.Simulation(
            cell_size=mp.Vector3(Sx, Sy2),
            boundary_layers=pml_layers,
            # k_point=kpoint,
            geometry=geometry,
            sources=source,
            default_material=Air,
            symmetries=[mp.Mirror(direction=mp.X)] if symmetry else None,
            resolution=resolution,
        )
        src = mp.GaussianSource(frequency=freq, fwidth=fwidth)
        source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
        opt.sim.change_sources(source)

        opt.sim.run(until=200)
        plt.figure(figsize=(Sx, Sy2))
        opt.sim.plot2D(fields=mp.Ez)
        fileName = f"./" + scriptName + "/" + scriptName_i + "/fieldAtWavelength" + str(int(100/freq) / 100) + ".png"
        plt.savefig(fileName)
        # try:
        #     Efield = opt.get_efield_z()
        #     print(Efield)
        #     plt.figure()
        #     plt.imshow(np.abs(Efield)**2, interpolation="nearest", origin="upper")
        #     plt.colorbar()
        #     fileName = f"./" + scriptName + "/" + scriptName_i + "/intensityAtWavelength" + str(1 / freq) + ".png"
        #     plt.savefig(fileName)
        # except:
        #     print("Plotting intensity failed, needs updated meep files")


    plt.figure()
    plt.plot(evaluation_history, "o-")
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("FOM")
    fileName = f"./" + scriptName + "/" + scriptName_i + "/FOM.png"
    plt.savefig(fileName)
    plt.close()


    estimatedSimulationTime = (datetime.datetime.now() - start0) * num_samples / (sample_nr + 1)
    sendNotification("Opitimization " + str(int(10000 * (sample_nr + 1) / num_samples)/100) + " % completed; eta at " +
                     str(start0 + estimatedSimulationTime))

print(best_nr)
print(best_f0)

lb = np.min(f0s, axis=1)
ub = np.max(f0s, axis=1)
mean = np.mean(f0s, axis=1)

plt.figure()
plt.fill_between(np.arange(num_samples), ub, lb, alpha=0.3)
plt.plot(mean, "o-")
plt.grid(True)
plt.xlabel("Sample #")
plt.ylabel("FOM")
fileName = f"./" + scriptName + "/FOM.png"
plt.savefig(fileName)
plt.close()

with open("./" + scriptName + "/best_result.txt", 'w') as var_file:
    var_file.write("best_nr \t" + str(best_nr) + "\n")
    var_file.write("best_f0 \t" + str(best_f0) + "\n")
    var_file.write("run_time \t" + str(datetime.datetime.now() - start0) + "\n")
    var_file.write("best_design \t" + str(best_design) + "\n")


# simulate intensities on best design
scriptName_i = "sample_" + str(best_nr)
best_design_reshaped = np.reshape(best_design, [num_layers, Nx])
opt.update_design([mapping(best_design_reshaped[i, :], eta_i, cur_beta) for i in range(num_layers)])

Sy2 = 20
geometry.append(mp.Block(
    center=mp.Vector3(y=-(Sy2 / 2 + Sy / 2) / 2),
    size=mp.Vector3(x=Sx, y=(Sy2 / 2 - Sy / 2)),
    material=SiO2
))
efficiency = []
FWHM = []
for freq in frequencies:
    # simulate intensities

    src = mp.GaussianSource(frequency=freq, fwidth=fwidth)
    source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
    sim = mp.Simulation(resolution=resolution,
                        cell_size=mp.Vector3(Sx, Sy2),
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        k_point=kpoint,
                        sources=source,
                        symmetries=[mp.Mirror(direction=mp.X)] if symmetry else None)

    near_fields_focus = sim.add_dft_fields([mp.Ez], freq, 0, 1, center=mp.Vector3(y=focal_point),
                                           size=mp.Vector3(x=design_region_width))
    near_fields_before = sim.add_dft_fields([mp.Ez], freq, 0, 1,
                                            center=mp.Vector3(y=-(-half_total_height + source_pos) / 2),
                                            size=mp.Vector3(x=design_region_width))
    near_fields = sim.add_dft_fields([mp.Ez], freq, 0, 1, center=mp.Vector3(),
                                     size=mp.Vector3(x=Sx, y=Sy2))

    sim.run(until_after_sources=100)

    focussed_field = sim.get_dft_array(near_fields_focus, mp.Ez, 0)
    before_field = sim.get_dft_array(near_fields_before, mp.Ez, 0)
    scattered_field = sim.get_dft_array(near_fields, mp.Ez, 0)
    # near_field = sim.get_dft_array(near_fields, mp.Ez, 0)

    focussed_amplitude = np.abs(focussed_field) ** 2
    scattered_amplitude = np.abs(scattered_field) ** 2
    before_amplitude = np.abs(before_field) ** 2

    [xi, yi, zi, wi] = sim.get_array_metadata(dft_cell=near_fields_focus)
    [xj, yj, zj, wj] = sim.get_array_metadata(dft_cell=near_fields)

    # plot colormesh
    plt.figure(dpi=150)
    plt.pcolormesh(xj, yj, np.rot90(np.rot90(np.rot90(scattered_amplitude))), cmap='inferno', shading='gouraud',
                   vmin=0,
                   vmax=scattered_amplitude.max())
    plt.gca().set_aspect('equal')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')

    # ensure that the height of the colobar matches that of the plot
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    plt.tight_layout()
    fileName = f"./" + scriptName + "/" + scriptName_i + "/intensityMapAtWavelength" + str(
        int(100 / freq) / 100) + ".png"
    plt.savefig(fileName)

    # plot intensity around focal point
    plt.figure()
    plt.plot(xi, focussed_amplitude, 'bo-')
    plt.xlabel("x (μm)")
    plt.ylabel("field amplitude")
    fileName = f"./" + scriptName + "/" + scriptName_i + "/intensityOverLineAtWavelength" + str(
        int(100 / freq) / 100) + ".png"
    plt.savefig(fileName)
    sendPhoto(fileName)

    efficiency.append(focussing_efficiency(focussed_amplitude, before_amplitude))
    FWHM.append(get_FWHM(focussed_amplitude, xi))

    np.save("./" + scriptName + "/" + scriptName_i + "/intensity_at_focus_freq" + str(int(100 / freq) / 100),
            focussed_amplitude)


with open("./" + scriptName + "/best_result.txt", 'a') as var_file:
    var_file.write("focussing_efficiency \t" + str(efficiency) + "\n")
    var_file.write("FWHM \t" + str(FWHM) + "\n")
    var_file.write("run_time \t" + str(datetime.datetime.now() - start0) + "\n")

sendNotification("Simulation finished; best FOM: " + str(best_f0))