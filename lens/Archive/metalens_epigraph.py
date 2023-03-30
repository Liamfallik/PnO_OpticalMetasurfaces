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
import random
import datetime
import requests # send notifications

scriptName = "metalens_epigraph3"
sendNotifs = True

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

# checking if the directory demo_folder exist or not.
if not os.path.exists("./" + scriptName + "_img"):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs("./" + scriptName + "_img")

mp.verbosity(0) # amount of info printed during simulation

# Materials
Si = mp.Medium(index=3.4)
Air = mp.Medium(index=1.0)

# Dimensions
design_region_width = 15
design_region_height = 2

# Boundary conditions
pml_size = 1.0
seed = 240 # make sure starting conditions are random, but always the same. Change seed to change starting conditions
np.random.seed(seed)
resolution = 30

# System size
Sx = 2 * pml_size + design_region_width
Sy = 2 * pml_size + design_region_height + 5
cell_size = mp.Vector3(Sx, Sy)

# Frequencies
nf = 3 # Amount of frequencies studied
# frequencies = np.array([1 / 1.5, 1 / 1.55, 1 / 1.6])
frequencies = 1./np.linspace(1.5, 1.6, 3)

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

fcen = 1 / 1.55 # Middle frequency of source
width = 0.2 # Relative width of frequency
fwidth = width * fcen # Absolute width of frequency
source_center = [0, -(design_region_height / 2 + 1.5), 0] # Source 1.5 µm below lens
source_size = mp.Vector3(design_region_width, 0, 0) # Source covers width of lens
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth) # Gaussian source
source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]

# Amount of variables
Nx = int(design_region_resolution * design_region_width)
Ny = int(design_region_resolution * design_region_height)

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
    filtered_field = mpa.conic_filter( # remain minimum feature size
        x,
        filter_radius,
        design_region_width,
        design_region_height,
        design_region_resolution,
    )

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta) # Make binary

    projected_field = (
        npa.flipud(projected_field) + projected_field
    ) / 2  # left-right symmetry

    # interpolate to actual materials
    return projected_field.flatten()

# Geometry: all is design region, no fixed parts
geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),
    # mp.Block(center=design_region.center, size=design_region.size, material=design_variables, e1=mp.Vector3(x=-1))
    #
    # The commented lines above impose symmetry by overlapping design region with the same design variable. However,
    # currently there is an issue of doing that; instead, we use an alternative approach to impose symmetry.
    # See https://github.com/NanoComp/meep/issues/1984 and https://github.com/NanoComp/meep/issues/2093
]

# Set-up simulation object
kpoint = mp.Vector3()
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air,
    symmetries=[mp.Mirror(direction=mp.X)],
    resolution=resolution,
)

# Focus point, 15 µm beyond centre of lens
far_x = [mp.Vector3(0, 15, 0)]
NearRegions = [ # region from which fields at focus point will be calculated
    mp.Near2FarRegion(
        center=mp.Vector3(0, design_region_height / 2 + 1.5), # 1.5 µm above lens
        size=mp.Vector3(design_region_width, 0), # spans design region
        weight=+1, # field contribution is positive (real)
    )
]
FarFields = mpa.Near2FarFields(sim, NearRegions, far_x) # Far-field object
ob_list = [FarFields]

def J1(FF):
    return -npa.abs(FF[0, :, 2]) ** 2 # only first (only point), vector of frequencies, and third field (Ez)

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
plt.savefig(f"./" + scriptName + "_img/optimizationDesign.png")

# Gradient
evaluation_history = [] # Keep track of objective function evaluations
cur_iter = [0] # Iteration

def f(x, grad):
    t = x[0]  # "dummy" parameter
    v = x[1:]  # design parameters
    if grad.size > 0:
        grad[0] = 1
        grad[1:] = 0
    return t

def c(result, x, gradient, eta, beta):
    if cur_iter[0] != 0:
        estimatedSimulationTime = (datetime.datetime.now() - start) * totalIterations / cur_iter[0]
        print("Current iteration: {}; current eta: {}, current beta: {}".format(cur_iter[0], eta, beta)
              + "; " + str(100 * cur_iter[0] / totalIterations) + "% completed ; eta at " +
              str(start + estimatedSimulationTime))
    else:
        print("Current iteration: {}".format(cur_iter[0]))
    cur_iter[0] += 1

    t = x[0]  # dummy parameter
    v = x[1:]  # design parameters

    f0, dJ_du = opt([mapping(v, eta, beta)])  # compute objective and gradient

    # Backprop the gradients through our mapping function
    my_grad = np.zeros(dJ_du.shape)
    for k in range(opt.nf):
        my_grad[:, k] = tensor_jacobian_product(mapping, 0)(v, eta, beta, dJ_du[:, k])
        # Note that we now backpropogate the gradients at individual frequencies

    # Assign gradients
    if gradient.size > 0:
        gradient[:, 0] = -1  # gradient w.r.t. "t"
        gradient[:, 1:] = my_grad.T  # gradient w.r.t. each frequency objective

    result[:] = np.real(f0) - t

    evaluation_history.append(np.real(f0)) # add objective function evaluation to list

    plt.figure() # Plot current design
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
    plt.savefig(f"./" + scriptName + "_img/img" + str(cur_iter[0]) + ".png")

    # return np.real(f0)

# Create the animation
animate = Animate2D(
    fields=None,
    # realtime=True,
    eps_parameters={'contour': False, 'alpha': 1, 'frequency': 1/1.55},
    plot_sources_flag=False,
    plot_monitors_flag=False,
    plot_boundaries_flag=False,
    update_epsilon=True,  # required for the geometry to update dynamically
    nb=False         # True required if running in a Jupyter notebook
)
animateField = Animate2D(
    fields=mp.Ez,
    # realtime=True,
    eps_parameters={'contour': False, 'alpha': 1, 'frequency': 1/1.55},
    plot_sources_flag=True,
    plot_monitors_flag=True,
    plot_boundaries_flag=True,
    update_epsilon=True,  # required for the geometry to update dynamically
    nb=False         # True required if running in a Jupyter notebook
)
# This will trigger the animation at the end of each simulation
opt.step_funcs=[mp.at_end(animate), mp.at_end(animateField)]

# Method of moving  asymptotes
algorithm = nlopt.LD_MMA
n = Nx * Ny  # number of parameters

# Initial guess
# x = np.ones((n,)) * 0.5 # average everywhere
# x = np.random.rand(n)
file_path = filedialog.askopenfilename(filetypes=[("Numpy Files", "*.npy")], initialdir='\\wsl.localhost\\Ubuntu-22.04\\home\\willem\\PO_Nano_code')

with open(file_path, 'rb') as f:
    x = np.load(f)

# lower and upper bounds
lb = np.zeros((Nx * Ny,))
ub = np.ones((Nx * Ny,))

# insert dummy parameter bounds and variable
x = np.insert(x, 0, -1)  # our initial guess for the worst error
lb = np.insert(lb, 0, -np.inf) # dummy variable can be between - infinity
ub = np.insert(ub, 0, 0) # and 0 (c needs to be negative, as t > f_i)

cur_beta = 64 # 4
beta_scale = 2
num_betas = 3 # 6
update_factor = 10 # 12
totalIterations = num_betas * update_factor
ftol = 1e-5
start = datetime.datetime.now()
print("Opitimization started at " + str(start))
if sendNotifs:
    sendNotification("Opitimization started at " + str(start))
for iters in range(num_betas):
    solver = nlopt.opt(algorithm, n+1) # add dummy variable
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_max_objective(f)
    solver.set_maxeval(update_factor) # stop when 12 iterations or reached
    solver.set_ftol_rel(ftol) # or when we converged
    solver.add_inequality_mconstraint(
        lambda r, x, g: c(r, x, g, eta_i, cur_beta), np.array([1e-3] * nf)
    )
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale
    estimatedSimulationTime = (datetime.datetime.now() - start) * num_betas / (iters + 1)
    print("Current iteration: {}".format(cur_iter[0]) + "; " + str(100 * cur_iter[0] / totalIterations) +
          "% completed ; eta at " + str(start + estimatedSimulationTime))
    if sendNotifs:
        sendNotification("Opitimization " + str(100 * (iters + 1) / num_betas) + " % completed; eta at " +
                        str(start + estimatedSimulationTime))

# Plot final design
opt.update_design([mapping(x[1:], eta_i, cur_beta)])
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
plt.savefig(f"./" + scriptName + "_img/finalDesign.png")

# Plot FOM
lb = -np.min(evaluation_history, axis=1)
ub = -np.max(evaluation_history, axis=1)
mean = -np.mean(evaluation_history, axis=1)

num_iters = lb.size

plt.figure()
plt.fill_between(np.arange(num_iters), ub, lb, alpha=0.3)
plt.plot(mean, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("FOM")
plt.savefig(f"./" + scriptName + "_img/FOM.png")

# Check intensities in optimal design
f0, dJ_du = opt([mapping(x[1:], eta_i, cur_beta // 2)], need_gradient=False)
frequencies = opt.frequencies

intensities = np.abs(opt.get_objective_arguments()[0][0, :, 2]) ** 2

# Plot intensities
plt.figure()
plt.plot(1 / frequencies, intensities, "-o")
plt.grid(True)
plt.xlabel("Wavelength (microns)")
plt.ylabel("|Ez|^2 Intensities")
fileName = f"./" + scriptName + "_img/intensities.png"
plt.savefig(fileName)
if sendNotifs:
    sendPhoto(fileName)

animate.to_gif(fps=5, filename=f"./" + scriptName + "_img/animation.gif")
animateField.to_gif(fps=5, filename=f"./" + scriptName + "_img/animationField.gif")

# Plot fields
for freq in frequencies:
    opt.sim = mp.Simulation(
        cell_size=mp.Vector3(Sx, 40),
        boundary_layers=pml_layers,
        k_point=kpoint,
        geometry=geometry,
        sources=source,
        default_material=Air,
        resolution=resolution,
    )
    src = mp.ContinuousSource(frequency=freq, fwidth=fwidth)
    source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
    opt.sim.change_sources(source)

    opt.sim.run(until=200)
    plt.figure(figsize=(10, 20))
    opt.sim.plot2D(fields=mp.Ez)
    fileName = f"./" + scriptName + "_img/fieldAtWavelength" + str(1/freq) + ".png"
    plt.savefig(fileName)
    if sendNotifs:
        sendPhoto(fileName)

np.save("x", x)

plt.close()

if sendNotifs:
    sendNotification("Simulation finished")

