import meep as mp
import meep.adjoint as mpa
from meep import Animate2D
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
import nlopt
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import os
import datetime


# checking if the directory demo_folder
# exist or not.
if not os.path.exists("./optimize_filter_img"):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs("./optimize_filter_img")

print("---------------------- Running optimize_filter.py ----------------------")

mp.verbosity(0) # amount of info printed during simulation
seed = 240 # make sure starting conditions are random, but always the same. Change seed to change starting conditions
np.random.seed(seed)
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

# Dimensions
waveguide_width = 0.5
design_region_width = 2.5
design_region_height = 2.5
waveguide_length = 0.5
pml_size = 1.0
frequencies = 1 / np.linspace(1.5, 1.6, 10)
resolution = 20 # pixels per µm

# Feature size, filters and thresholds
minimum_length = 0.09  # minimum length scale [µm], features won't become smaller
eta_i = (
    0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
print(filter_radius)
design_region_resolution = int(1 * resolution)

# cell size: based on waveguides and design region
Sx = 2 * pml_size + 2 * waveguide_length + design_region_width
Sy = 2 * pml_size + design_region_height + 0.5
cell_size = mp.Vector3(Sx, Sy)

# Boundary conditions
pml_layers = [mp.PML(pml_size)]

# Source
fcen = 1 / 1.55 # frequency
width = 0.1 # relative pulse width (Gaussian)
fwidth = width * fcen
source_center = [-Sx / 2 + pml_size + waveguide_length / 3, 0, 0]
source_size = mp.Vector3(0, Sy, 0) # source over complete simulation width
kpoint = mp.Vector3(1, 0, 0) # direction of wave
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
source = [
    mp.EigenModeSource(
        src,
        eig_band=1,
        direction=mp.NO_DIRECTION,
        eig_kpoint=kpoint,
        size=source_size,
        center=source_center,
    )
]

# Design region
Nx = int(design_region_resolution * design_region_width)
Ny = int(design_region_resolution * design_region_height)

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables, volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(design_region_width, design_region_height, 0)) # 1x1 volume in the center
)

# Boundary
x_g = np.linspace(-design_region_width / 2, design_region_width / 2, Nx) # x-points
y_g = np.linspace(-design_region_height / 2, design_region_height / 2, Ny) # y-points
X_g, Y_g = np.meshgrid(x_g, y_g, sparse=True, indexing="ij") # grid

left_wg_mask = (X_g == -design_region_width / 2) & (np.abs(Y_g) <= waveguide_width / 2) # connection with left wave-guide
top_wg_mask = (Y_g == design_region_width / 2) & (np.abs(X_g) <= waveguide_width / 2) # connection with top wave-guide
Si_mask = left_wg_mask | top_wg_mask # total Si boundary

border_mask = (
    (X_g == -design_region_width / 2)
    | (X_g == design_region_width / 2)
    | (Y_g == -design_region_height / 2)
    | (Y_g == design_region_height / 2) # total boundary
)
SiO2_mask = border_mask.copy()
SiO2_mask[Si_mask] = False # SiO2 boundary is total boundary excluding Si boundary

# Filters
def mapping(x, eta, beta):
    x = npa.where(Si_mask.flatten(), 1, npa.where(SiO2_mask.flatten(), 0, x)) # boundary conditions: Si at Si boundary
    # and SiO2 at SiO2 boundary

    # filter
    filtered_field = mpa.conic_filter(
        x,
        filter_radius,
        design_region_width,
        design_region_height,
        design_region_resolution,
    ) # filter for minimum feature size

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta) # make closer to binary

    projected_field = (
        npa.rot90(projected_field.T, 2) + projected_field
    ) / 2  # 90-degree symmetry

    # interpolate to actual materials
    return projected_field.flatten() # create 1D array, instead of 2D

# Geometry
geometry = [
    mp.Block(
        center=mp.Vector3(x=-Sx / 4), material=Si, size=mp.Vector3(Sx / 2, 0.5, 0)
    ),  # horizontal waveguide
    mp.Block(
        center=mp.Vector3(y=Sy / 4), material=Si, size=mp.Vector3(0.5, Sy / 2, 0)
    ),  # vertical waveguide
    mp.Block(
        center=design_region.center, size=design_region.size, material=design_variables
    ),  # design region
    # mp.Block(center=design_region.center, size=design_region.size, material=design_variables,
    #        e1=mp.Vector3(x=-1).rotate(mp.Vector3(z=1), np.pi/2), e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), np.pi/2))
    #
    # The commented lines above impose symmetry by overlapping design region with the same design variable. However,
    # currently there is an issue of doing that; We give an alternative approach to impose symmetry in later tutorials.
    # See https://github.com/NanoComp/meep/issues/1984 and https://github.com/NanoComp/meep/issues/2093
]

# Simulation object
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    # eps_averaging=False,
    default_material=SiO2,
    resolution=resolution,
)

# Measurements
mode = 1
    # After bend
TE_top = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(0, Sx / 2 - pml_size - 2 * waveguide_length / 3, 0), size=mp.Vector3(x=Sx)), mode=mode
)
    # Before bend
TE0 = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(x=-Sx / 2 + pml_size + 2 * waveguide_length / 3), size=mp.Vector3(y=Sy)), mode=mode
)
ob_list = [TE0, TE_top]

def J(before, after):
    return npa.mean(npa.abs(after/before) ** 2) # average over all frequencies

# Optimization problem definition
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J, # Take the square of the magnitude
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies, # multiple frequencies between 1.5 and 1.6: J will take average
)


# Initialize design region randomly
x0 = np.random.rand(Nx * Ny)
opt.update_design([x0])

# Choose Method of Moving Asymptotes as optimization algorithm
algorithm = nlopt.LD_MMA
n = Nx * Ny # total amount of variables

# Keep track of objective function evaluations
evaluation_history = [] # Saves objective function evaluation
cur_iter = [0] # Saves iteration

def f(v, gradient, cur_beta):
    if cur_iter[0] != 0:
        estimatedSimulationTime = (datetime.datetime.now() - start) * totalIterations / cur_iter[0]
        # SecondsToGo = (time.time() - start) * (totalIterations - cur_iter[0]) / cur_iter[0]
        print("Current iteration: {}".format(cur_iter[0]) + "; " + str(100 * cur_iter[0] / totalIterations) +
              "% completed ; eta at " + str(start + estimatedSimulationTime))
              # " hours, " + str((SecondsToGo % 3600) // 60) + " minutes and " + str(SecondsToGo % 60) + " seconds.")
    else:
        print("Current iteration: {}".format(cur_iter[0]))
    cur_iter[0] += 1

    f0, dJ_du = opt([mapping(v, eta_i, cur_beta)])  # compute objective and gradient

    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)( # Jacobian of the mapping multiplied by v; 0 for the adjoint method
            v, eta_i, cur_beta, np.sum(dJ_du, axis=1)
        )  # backprop

    evaluation_history.append(np.real(f0))

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

# Initial guess
x = np.ones((n,)) * 0.5 # start with 0.5 everywhere
x[Si_mask.flatten()] = 1  # set the edges of waveguides to silicon
x[SiO2_mask.flatten()] = 0  # set the other edges to SiO2

# lower and upper bounds
lb = np.zeros((Nx * Ny,)) # lower bound is 0 everywhere
lb[Si_mask.flatten()] = 1 # except at the Si boundaries
ub = np.ones((Nx * Ny,)) # upper bound is 1 everywhere
ub[SiO2_mask.flatten()] = 0 # except at the SiO2 boundary

cur_beta = 4 # we start with beta = 4
beta_scale = 2 # and multiply by 2 from time to time, to come closer to binary solution
num_betas = 6 # we'll multiply it 5 times, so go up to 4*2^5 = 128
update_factor = 12 # 12 iterations before we change beta
totalIterations = update_factor * num_betas
# start = time.time() # save start time
start = datetime.datetime.now()
print("Opitimization started at " + str(start))
for iters in range(num_betas):
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb) # a bit more sophisticated than just 0
    solver.set_upper_bounds(ub) # a bit more sophisticated than just 1
    solver.set_max_objective(lambda a, g: f(a, g, cur_beta)) # use lambda function to make sure beta can change everytime
    solver.set_maxeval(update_factor) # do 12 iterations
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale


# Plot figure of merit (FOM)
plt.figure()
plt.plot(np.array(evaluation_history)*100, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Transmission [%]")
plt.savefig(f'./optimize_filter_img/FOM.png')

animate.to_gif(fps=5, filename=f'./optimize_filter_img/animation.gif')
animateField.to_gif(fps=5, filename=f'./optimize_filter_img/animationField.gif')

# Forward run
f0, dJ_du = opt([mapping(x, eta_i, cur_beta // 2)], need_gradient=False)
frequencies = opt.frequencies
source_coef, top_coef = opt.get_objective_arguments()

top_profile = np.abs(top_coef / source_coef) ** 2

plt.figure()
plt.plot(1 / frequencies, top_profile * 100, "-o")
plt.grid(True)
plt.xlabel("Wavelength (microns)")
plt.ylabel("Transmission (%)")
# plt.ylim(98,100)
plt.savefig(f'./optimize_filter_img/TransmissionFinal.png')

# Final simulation

print("----------- Starting simulation -----------")

src = mp.ContinuousSource(frequency=1 / 1.55, fwidth=fwidth)
source2 = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]

opt.sim = mp.Simulation(
    cell_size=mp.Vector3(Sx, Sy),
    boundary_layers=pml_layers,
    k_point=kpoint,
    geometry=geometry,
    sources=source2,
    # default_material=Si,
    resolution=resolution,
)

opt.sim.run(until=200)
plt.figure()
opt.sim.plot2D(fields=mp.Ez)
plt.savefig("./optimize_filter_img/Ez.png")

print("----------- Simulation completed -----------")
# Save data
print("----------- Saving data -----------")

eps_data = opt.sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Dielectric)
np.save("./optimize_filter_img/epsilon", eps_data)
ez_data = opt.sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)
np.save("./optimize_filter_img/Ez", ez_data)