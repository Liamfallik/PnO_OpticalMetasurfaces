# Import meep and autograd, a widely used automatic differentiation package.
# autograd wraps around numpy and allows us to quickly differentiate functions
# composed of standard numpy routines.
# This will be especially useful when we want to formulate our objective function.

import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
import nlopt
from matplotlib import pyplot as plt
import imageio
import os
from timer import Timer


# checking if the directory demo_folder
# exist or not.
if not os.path.exists("./Intro_img"):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs("./Intro_img")


mp.verbosity(0)
seed = 240
np.random.seed(seed)
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

# specify the domain of our waveguide bend simulation

resolution = 20

Sx = 6
Sy = 5
cell_size = mp.Vector3(Sx, Sy)

pml_layers = [mp.PML(1.0)]

# Define the sources.
# We'll use a narrowband Gaussian pulse, even though our objective function will
# only operate over a single wavelength (for this example). While we could use
# the CW solver, it's often faster (and more dependable) to use the timestepping
# routines and a narrowband pulse.

fcen = 1 / 1.55
width = 0.1
fwidth = width * fcen
source_center = [-1, 0, 0]
source_size = mp.Vector3(0, 2, 0)
kpoint = mp.Vector3(1, 0, 0)
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

# Define our waveguide geometry and "design region".
# The design region takes a 10x10 grid of randomly generated design variables
# and maps them to a point within the specified volume. Since meep operates under
# the "illusion of continuitiy", it's important we provide an interpolator to
# perform this mapping.
#
# In this case, we'll use a builtin bilinear interpolator to take care of this for
# us. You can use whatever interpolation function you want, provided it can return
# either a medium or permittivity (as described in the manual) and you can calculate
# the gradient of the interpolator with respect to the inputs (often just a matrix
# multiplication). The built-in interpolator takes care of all of this for us.

design_region_resolution = 10
Nx = design_region_resolution
Ny = design_region_resolution

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables, volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(1, 1, 0))
)


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

# Formulate the simulation object

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    eps_averaging=False,
    resolution=resolution,
)

# defining our objective function. Objective functions must be composed of "field functions" that
# transform the recorded fields. Right now, only the Eigenmode Decomposition monitors are readily
# accessible from the adjoint API. That being said, it is easy to extend the API to other field
# functions, like Poynting fluxes.
#
# In our case, we just want to maximize the power in the fundamental mode at the top of the waveguide
# bend. We'll define a new object that specifies these details. We'll also create a list of our
# objective "quantities" (just one in our case).

TE0 = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(0, 1, 0), size=mp.Vector3(x=2)), mode=1
)
ob_list = [TE0]

# Our objective function will take the magnitude squared of our predefined objective quantity.
# We'll define the function as a normal python function. We'll use dummy parameters that map sequentially
# to the list of objective quantities we defined above. We'll also use autograd's version of numpy
# so that the adjoint solver can easily differentiate our objective function with respect to each of
# these quantities.

def J(alpha):
    return npa.abs(alpha) ** 2

# We can now define an OptimizationProblem using our simulation object, objective function,
# and objective quantities (or arguments). We'll also tell the solver to examine the Ez component
# of the Fourier transformed fields. The solver will stop the simulation after these components
# have stopped changing by a certain relative threshold (default is 1e-6).

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=ob_list,
    design_regions=[design_region],
    fcen=fcen,
    df=0,
    nf=1,
)

# We will now initialize our object with a set of random design parameters. We'll use numpy's
# random library to generate a uniform random variable between 1 and 12, to correspond to the
# refractive index of air and silicon.

x0 = np.random.rand(Nx * Ny)
opt.update_design([x0])

# Visualize our final simulation domain with any extra monitors as defined by our objective
# function. This plot2D function is just like Simulation's plot2D, only it takes an additional
# first argument. We'll set it to True to tell the solver to initialize the solver and clear
# any stored fields.

# opt.plot2D(True, frequency=1 / 1.55)
# plt.show()

# Calculate the gradient and the cost function evaluation. We do so by calling our solver
# object directly. The object returns the objective function evaluation, f0, and the
# gradient, dJ_du.

f0, dJ_du = opt()

# We can visualize these gradients.

plt.figure()
plt.imshow(np.rot90(dJ_du.reshape(Nx, Ny)))
plt.savefig(f'./Intro_img/gradient.png')

print("Time spent on Fourier transforming:")
print(mp.Simulation.time_spent_on(sim,6))
print("Time spent on near-to-far-field transform:")
print(mp.Simulation.time_spent_on(sim,8))

# timr = Timer()
# timr.start()
# timr.stop()

# #To verify the accuracy of our method, we'll perform a finite difference approximation.
#
# #Luckily, our solver has a built finite difference method (calculate_fd_gradient).
# #Since the finite difference approximates require several expensive simulations, we'll
# # only estimate 20 of them (randomly sampled by our function).
#
# db = 1e-3
# choose = 10
# g_discrete, idx = opt.calculate_fd_gradient(num_gradients=choose, db=db)
#
# # Compare the results by fitting a line to the two gradients
#
# (m, b) = np.polyfit(np.squeeze(g_discrete), dJ_du[idx], 1)
#
# # Plot the results
#
# min_g = np.min(g_discrete)
# max_g = np.max(g_discrete)
#
# fig = plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.plot([min_g, max_g], [min_g, max_g], label="y=x comparison")
# plt.plot([min_g, max_g], [m * min_g + b, m * max_g + b], "--", label="Best fit")
# plt.plot(g_discrete, dJ_du[idx], "o", label="Adjoint comparison")
# plt.xlabel("Finite Difference Gradient")
# plt.ylabel("Adjoint Gradient")
# plt.legend()
# plt.grid(True)
# plt.axis("square")
#
# plt.subplot(1, 2, 2)
# rel_err = (
#     np.abs(np.squeeze(g_discrete) - np.squeeze(dJ_du[idx]))
#     / np.abs(np.squeeze(g_discrete))
#     * 100
# )
# plt.semilogy(g_discrete, rel_err, "o")
# plt.grid(True)
# plt.xlabel("Finite Difference Gradient")
# plt.ylabel("Relative Error (%)")
#
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.suptitle("Resolution: {} Seed: {} Nx: {} Ny: {}".format(resolution, seed, Nx, Ny))
# plt.show()
#
# # We notice strong agreement between the adjoint gradients and the finite difference
# # gradients. Let's bump up the resolution to see if the results are consistent.
#
# # resolution = 30
# # opt.sim.resolution = resolution
# # f0, dJ_du = opt()
# # g_discrete, idx = opt.calculate_fd_gradient(num_gradients=choose, db=db)











