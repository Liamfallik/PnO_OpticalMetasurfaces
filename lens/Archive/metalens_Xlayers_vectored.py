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

num_layers = 2
scriptName = "metalens_Xlayers_vectored_Air-NbOx_img"

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
design_region_width =  [4] * num_layers
design_region_height = [0.3] * num_layers
half_total_heigth = sum(design_region_height) / 2
empty_space = 0

# Boundary conditions
pml_size = 1.0
seed = 240 # make sure starting conditions are random, but always the same. Change seed to change starting conditions
np.random.seed(seed)
resolution = 50 # 50 > 49

# System size
Sx = 2 * pml_size + max(design_region_width) + 2 * empty_space
Sy = 2 * pml_size + sum(design_region_height) + 2 # + 5
cell_size = mp.Vector3(Sx, Sy)

# Frequencies
nf = 3 # Amount of frequencies studied
# frequencies = np.array([1 / 1.5, 1 / 1.55, 1 / 1.6])
frequencies = 1./np.linspace(0.55, 0.65, 3)

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

fcen = 1 / 0.6 # Middle frequency of source
width = 0.2 # Relative width of frequency
fwidth = width * fcen # Absolute width of frequency
source_center = [0, -(half_total_heigth + 0.75), 0] # normally 1.5 instead of 0.75 # Source 1.5 µm below lens
source_size = mp.Vector3(max(design_region_width) + 2*empty_space, 0, 0) # Source covers width of lens
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth) # Gaussian source
source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]

# Amount of variables
Nx = [int(design_region_resolution * width_design) for width_design in design_region_width]
Ny = [int(1) for j in range(num_layers)] # int(design_region_resolution * design_region_height)


design_variables = [mp.MaterialGrid(mp.Vector3(Nx[i], Ny[i]), Air, NbOx, grid_type="U_MEAN") for i in range(num_layers)]
print(design_variables)
design_regions = [mpa.DesignRegion(
        design_variables[i],
        volume=mp.Volume(
            center=mp.Vector3(y = -half_total_heigth + sum(design_region_height[:i]) + design_region_height[i] / 2),
            size=mp.Vector3(design_region_width[i], design_region_height[i], 0),
        ),
    ) for i in range(num_layers)]
print(design_regions)

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


    # filtered_x = np.zeros([Nx, Ny*num_layers])
    filtered_x = []
    done = 0
    for i in range(num_layers):
        xv = np.arange(0, Lx[i] / 2, Lx[i] / Nx[i])  # Lx / Nx instead of 1 / resolution
        yv = np.arange(0, Ly[i] / 2, Ly[i] / Ny[i])

        X, Y = np.meshgrid(xv, yv, sparse=True, indexing="ij")
        h = np.where(
            X ** 2 + Y ** 2 < radius ** 2, (1 - np.sqrt(abs(X ** 2 + Y ** 2)) / radius), 0
        )

        step = Nx[i]*Ny[i]
        xi = x[done:done+step].reshape(Nx[i], Ny[i]) # Select a design region and ensure the input is 2D
        done += step
        # filtered_x[:, i*Ny:(i+1)*Ny] = mpa.simple_2d_filter(xi, h) # apply filter
        filtered_x.append(mpa.simple_2d_filter(xi, h)) # apply filter
    return filtered_x

# Filter and projection
def mapping(x, eta, beta, flat=False):

    # filter
    filtered_fields = conic_filter2( # remain minimum feature size
        x,
        filter_radius,
        design_region_width,
        design_region_height,
        Nx,
        Ny
    )

    filt_and_proj_fields = []
    for filtered_field in filtered_fields:
        # projection
        projected_field = mpa.tanh_projection(filtered_field, beta, eta) # Make binary

        projected_field = (
            npa.flipud(projected_field) + projected_field
        ) / 2  # left-right symmetry
        # print(flat)
        if flat:
            filt_and_proj_fields.extend(projected_field.flatten())
            # print(np.shape(filt_and_proj_fields))
        else:
            filt_and_proj_fields.append(projected_field.flatten())

    return filt_and_proj_fields

# Geometry: all is design region, no fixed parts
geometry = [
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_region.design_parameters
    ) for design_region in design_regions
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

# Focus point, 7.5 µm beyond centre of lens
far_x = [mp.Vector3(0, 7.5, 0)] # used to be 15
NearRegions = [ # region from which fields at focus point will be calculated
    mp.Near2FarRegion(
        center=mp.Vector3(0, half_total_heigth + 0.75), # normally 1.5 instead of 0.75 # 1.5 µm above lens
        size=mp.Vector3(max(design_region_width) + 2*empty_space, 0), # spans design region
        weight=+1, # field contribution is positive (real)
    )
]
FarFields = mpa.Near2FarFields(sim, NearRegions, far_x) # Far-field object
ob_list = [FarFields]

def J1(FF):
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
        print("Current iteration: {}".format(cur_iter[0]) + "; " + str(100 * cur_iter[0] / totalIterations) +
              "% completed ; eta at " + str(start + estimatedSimulationTime))
    else:
        print("Current iteration: {}".format(cur_iter[0]))
    cur_iter[0] += 1


    filt_proj_field = mapping(v, eta_i, cur_beta)
    if np.array(filt_proj_field).ndim > 1:
        f0, dJ_du = opt(filt_proj_field)
    else:
        f0, dJ_du = opt([filt_proj_field])

    # flatten: merge all design regions in 1 large vector
    if np.array(dJ_du).ndim == 3:
        dJ_du_shape = np.shape(dJ_du)
        dJ_du = np.reshape(dJ_du, [dJ_du_shape[0]*dJ_du_shape[1], dJ_du_shape[2]])
    print(dJ_du)
    if gradient.size > 0:
        # print("shape v = " + str(np.shape(v)))
        # print("shape dJ_du = " + str(np.shape(dJ_du)))
        # sgfd = np.sum(dJ_du, axis=1)
        # print(sgfd)
        # print(np.shape(sgfd))

        gradient[:] = tensor_jacobian_product(lambda x, eta, beta: mapping(x, eta, beta, flat=True), 0)(
            v, eta_i, cur_beta, np.sum(dJ_du, axis=1)
        )  # backprop
    print(gradient)
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
    plt.savefig("./" + scriptName + "/img_{" + str(cur_iter[0]) + ".png")

    return np.real(f0)

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
n = [int(Nx[j] * Ny[j]) for j in range(num_layers)]  # number of parameters

# Initial guess
# x = np.ones((n,)) * 0.5 # average everywhere
# x = [np.random.rand(n[j]) for j in range(num_layers)]
x = np.random.rand(sum(n))

# lower and upper bounds
# lb = [np.zeros((n[j],)) for j in range(num_layers)]
# ub = [np.ones((n[j],)) for j in range(num_layers)]
lb = np.zeros(sum(n))
ub = np.ones(sum(n))

cur_beta = 4
beta_scale = 2
num_betas = 6
update_factor = 12
totalIterations = num_betas * update_factor
ftol = 1e-5
start = datetime.datetime.now()
print(scriptName + "- opitimization started at " + str(start))
sendNotification("Opitimization started at " + str(start))
for iters in range(num_betas):
    solver = nlopt.opt(algorithm, sum(n)) #sum or not?
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
    solver.set_maxeval(update_factor) # stop when 12 iterations or reached
    solver.set_ftol_rel(ftol) # or when we converged
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale
    estimatedSimulationTime = (datetime.datetime.now() - start) * num_betas / (iters + 1)
    print("Current iteration: {}".format(cur_iter[0]) + "; " + str(100 * cur_iter[0] / totalIterations) +
          "% completed ; eta at " + str(start + estimatedSimulationTime))
    sendNotification("Opitimization " + str(100 * (iters + 1) / num_betas) + " % completed; eta at " +
                        str(start + estimatedSimulationTime))
    np.save("./" + scriptName + "/x", x)
    print(x)

# Plot final design
print(x)
opt.update_design([mapping(x, eta_i, cur_beta)])
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

animate.to_gif(fps=5, filename="./" + scriptName + "/animation.gif")
animateField.to_gif(fps=5, filename="./" + scriptName + "/animationField.gif")


# Plot fields
for freq in frequencies:
    opt.sim = mp.Simulation(
        cell_size=mp.Vector3(Sx, 20),
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
    fileName = f"./" + scriptName + "/fieldAtWavelength" + str(1/freq) + ".png"
    plt.savefig(fileName)

plt.close()

sendNotification("Simulation finished")
