import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
import nlopt
from matplotlib import pyplot as plt
import imageio
import os

# checking if the directory demo_folder
# exist or not.
if not os.path.exists("./optimize_bend_img"):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs("./optimize_bend_img")
if not os.path.exists("./img"):
    os.makedirs("./img")

print("---------------------- Running optimize_bend.py ----------------------")

mp.verbosity(0) # amount of info printed during simulation
seed = 241 # make sure starting conditions are random, but always the same. Change seed to change starting conditions
np.random.seed(seed)
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.44)

# pixels per µm
resolution = 20

# cell size
Sx = 6
Sy = 5
cell_size = mp.Vector3(Sx, Sy)

# Boundary conditions
pml_layers = [mp.PML(1.0)]

# Source
fcen = 1 / 1.55 # frequency
width = 0.1 # pulse width
fwidth = width * fcen
source_center = [-1.5, 0, 0]
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

# Design region
design_region_resolution = 10 # 10 pixels in each dimension
Nx = design_region_resolution
Ny = design_region_resolution

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), SiO2, Si, grid_type="U_MEAN")
design_region = mpa.DesignRegion(
    design_variables, volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(1, 1, 0)) # 1x1 volume in the center
)

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
    eps_averaging=False,
    resolution=resolution,
)

# Measurements
    # After bend
TE_top = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(0, 1, 0), size=mp.Vector3(x=2)), mode=1
)
    # Before bend
TE0 = mpa.EigenmodeCoefficient(
    sim, mp.Volume(center=mp.Vector3(-1, 0, 0), size=mp.Vector3(y=2)), mode=1
)
ob_list = [TE0, TE_top]

def J(before, after):
    return npa.abs(after/before) ** 2

# Optimization problem definition
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J, # Take the square of the magnitude
    objective_arguments=ob_list,
    design_regions=[design_region],
    fcen=fcen, # given frequency to measure
    df=0, # variation in frequency
    nf=1, # amount of frequencies given
)

# Initialize design region randomly
x0 = np.random.rand(Nx * Ny)
opt.update_design([x0])

# Show initial conditions
opt.plot2D(True)
plt.axis("off")
plt.savefig(f'./optimize_bend_img/initial conditions.png')
plt.figure()
opt.update_design([x0])
opt.plot2D(
    True,
    plot_monitors_flag=False,
    output_plane=mp.Volume(center=(0, 0, 0), size=(2, 2, 0)),
)
plt.savefig(f'./img/img_{0}.png',
            transparent=False,
            facecolor='white'
            )

# Call objective and gradient
f0, dJ_du = opt()

# Plot gradient
plt.figure()
plt.imshow(np.rot90(dJ_du.reshape(Nx, Ny)))
plt.savefig(f'./optimize_bend_img/gradient.png')

# Choose Method of Moving Asymptotes as optimization algorithm
algorithm = nlopt.LD_MMA
n = Nx * Ny # total amount of variables
maxeval = 10 # max 10 steps

# Keep track of objective function evaluations
evaluation_history = [] # Saves objective function evaluation
sensitivity = [0] # Saves gradient
counter = [1]

def f(x, grad):
    print("Run " + str(counter[0]))
    f0, dJ_du = opt([x])
    f0 = f0[0] # f0 is an array of length 1
    if grad.size > 0:
        grad[:] = np.squeeze(dJ_du)
    evaluation_history.append(np.real(f0))
    sensitivity[0] = dJ_du
    # Plot current position
    plt.figure()
    opt.update_design([x])
    opt.plot2D(
        True,
        plot_monitors_flag=False,
        output_plane=mp.Volume(center=(0, 0, 0), size=(2, 2, 0)),
    )
    plt.axis("off")
    plt.savefig(f'./img/img_{counter[0]}.png',
                transparent=False,
                facecolor='white'
                )
    counter[0] += 1
    return np.real(f0)

# define solver object
solver = nlopt.opt(algorithm, n)
solver.set_lower_bounds(0)
solver.set_upper_bounds(1) # values between 0 and 1 (between SiO2 and Si)
solver.set_max_objective(f) # maximize and not minimize objective
solver.set_maxeval(maxeval) # max amount of steps
x = solver.optimize(x0) # run optimization

# Plot figure of merit (FOM)
plt.figure()
plt.plot(np.array(evaluation_history)*100, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Transmission (%)")
plt.savefig(f'./optimize_bend_img/FOM.png')

# Plot best design
opt.update_design([x])
opt.plot2D(
    True,
    plot_monitors_flag=False,
    output_plane=mp.Volume(center=(0, 0, 0), size=(2, 2, 0)),
)
plt.axis("off")
plt.savefig(f'./img/img_{maxeval+1}.png')


# Make gif of steps
frames = []
for t in range(maxeval+2):
    image = imageio.v2.imread(f'./img/img_{t}.png')
    frames.append(image)

imageio.mimsave('./optimize_bend_img/example.gif', # output gif
                frames,          # array of input frames
                fps = 1)         # optional: frames per second


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
plt.savefig("./optimize_bend_img/Ez.png")

print("----------- Simulation completed -----------")
# Save data
print("----------- Saving data -----------")

eps_data = opt.sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Dielectric)
np.save("./optimize_bend_img/epsilon", eps_data)
ez_data = opt.sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)
np.save("./optimize_bend_img/Ez", ez_data)

# Plot data
# print("----------- Plotting data -----------")

# plt.figure()
# plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
# plt.axis('off')
# plt.savefig("./optimize_bend_img/Epsilon.png")
#
# plt.figure()
# plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
# plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
# plt.axis('off')
# plt.savefig("./optimize_bend_img/Ez.png")
# plt.show()

