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
scriptName = "Retroreflector_Test6_1_S5_S15_Half_Symm" # Optimization of two layers, multiple angles
symmetry = False # Impose symmetry around x = 0 line
#
# def sendNotification(message):
#     token = "5421873058:AAFIKUk8fSksmo2qe9rHZ0dmYo0CI12fYyU"
#     myuserid = 6297309186
#     method = '/sendMessage'
#     url = f"https://api.telegram.org/bot{token}"
#     params = {"chat_id": myuserid, "text": message}
#     try:
#         r = requests.get(url + method, params=params)
#     except:
#         print("No internet connection: couldn't send notification...")
#
# def sendPhoto(image_path):
#     token = "5421873058:AAFIKUk8fSksmo2qe9rHZ0dmYo0CI12fYyU"
#     myuserid = 6297309186
#     data = {"chat_id": myuserid}
#     url = f"https://api.telegram.org/bot{token}" + "/sendPhoto"
#     with open(image_path, "rb") as image_file:
#         try:
#             ret = requests.post(url, data=data, files={"photo": image_file})
#             return ret.json()
#         except:
#             print("No internet connection: couldn't send notification...")


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

    if True:
        filtered_field = (
            npa.flipud(filtered_field) + filtered_field
        ) / 2  # left-right symmetry

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)  # Make binary

    # interpolate to actual materials
    return projected_field.flatten()

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
num_layers = 2 # amount of layers
design_region_width = 10 # width of layer
design_region_height = [0.24,0.24/2] # height of layer
spacing = 4 # spacing between layers
half_total_height = sum(design_region_height) / 2 + (num_layers - 1) * spacing / 2
empty_space = 0 # free space in simulation left and right of layer

# Boundary conditions
pml_size = 1.0 # thickness of absorbing boundary layer

resolution = 50 # 50 --> amount of grid points per µm; needs to be > 49 for TiOx and 0.55 µm

# System size
space_below = 1.7 # includes PML
Sx = 2 * pml_size + design_region_width + 2 * empty_space
Sy = half_total_height * 2 + space_below
cell_size = mp.Vector3(Sx, Sy)



# Frequencies
nf = 1 # Amount of frequencies studied
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
pml_layers = [mp.PML(pml_size, direction=mp.X), mp.PML(pml_size, direction=mp.Y, side=1)]

# Source
fcen = frequencies[0]
fwidth = 0.03 # 0.2
# Set-up simulation object
rot_angle = np.radians(15)
kpoint = mp.Vector3(y=1).rotate(mp.Vector3(z=1), -rot_angle)
# kpoint = mp.Vector3(1,1,0)
source_center = [0, -(half_total_height - space_below / 2 + 0.4), 0] # Source 1 µm below lens
source_size = mp.Vector3(design_region_width, 0, 0) # Source covers width of lens
# src = mp.GaussianSource(frequency=fcen, fwidth=fwidth) # Gaussian source
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
srcs = [mp.GaussianSource(frequency=fcen, fwidth=fwidth) for fcen in frequencies] # Gaussian source
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center) for src in srcs]
source = [mp.EigenModeSource(
        src,
        eig_band=1,
        direction=mp.NO_DIRECTION,
        eig_kpoint=kpoint,
        eig_parity=mp.ODD_Z,
        size=source_size,
        center=source_center,
    ) for src in srcs]

rot_angle2 = np.radians(5)
kpoint2 = mp.Vector3(y=1).rotate(mp.Vector3(z=1), -rot_angle2)
# source_center = [0, -(half_total_height - space_below / 2 + 0.4), 0] # Source 1 µm below lens
# source_size = mp.Vector3(design_region_width, 0, 0) # Source covers width of lens
# srcs = [mp.GaussianSource(frequency=fcen, fwidth=fwidth) for fcen in frequencies] # Gaussian source
source2 = [mp.EigenModeSource(
        src,
        eig_band=1,
        direction=mp.NO_DIRECTION,
        eig_kpoint=kpoint2,
        eig_parity=mp.ODD_Z,
        size=source_size,
        center=source_center,
    ) for src in srcs]

# Amount of variables
Nx = int(design_region_resolution * design_region_width)
Ny = 1 # int(design_region_resolution * design_region_height)


design_variables = [mp.MaterialGrid(mp.Vector3(Nx), SiO2, TiOx, grid_type="U_MEAN") for i in range(num_layers)] # SiO2
design_regions = [mpa.DesignRegion(
    design_variables[i],
    volume=mp.Volume(
        center=mp.Vector3(y=-half_total_height + 0.5 * design_region_height[i] + sum(design_region_height[:i]) + i * spacing + space_below / 2),
        size=mp.Vector3(design_region_width, design_region_height[i], 0),
    ),
) for i in range(num_layers)]


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
        center = mp.Vector3(y=(Sy/2) - design_region_height[1] - (spacing/2)),
        size=mp.Vector3(x = Sx, y = spacing),
        material=SiO2
    ))

# # Set-up simulation object
# kpoint = mp.Vector3(1,1,0)

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air, # Air
    symmetries=[mp.Mirror(direction=mp.X)] if symmetry else None,
    resolution=resolution,
)
sim2 = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source2,
    default_material=Air, # Air
    symmetries=[mp.Mirror(direction=mp.X)] if symmetry else None,
    resolution=resolution,
)

# Focal point 1
focal_point = -100
far_x = [mp.Vector3(focal_point*np.tan(rot_angle), focal_point, 0)]
NearRegions = [ # region from which fields at focus point will be calculated
    mp.Near2FarRegion(
        center=mp.Vector3(0, -(half_total_height - space_below / 2 + 0.5)), # 0.4 µm above lens
        size=mp.Vector3(design_region_width + 2*empty_space, 0), # spans design region
        weight= -1, # field contribution is positive (real)
    )
]
FarFields = mpa.Near2FarFields(sim, NearRegions, far_x) # Far-field object
ob_list = [FarFields]

# Focal point 2
focal_point2 = -100
far_x2 = [mp.Vector3(focal_point2*np.tan(rot_angle2), focal_point2, 0)]
FarFields2 = mpa.Near2FarFields(sim2, NearRegions, far_x2) # Far-field object
ob_list2 = [FarFields2]

def J1(FF):
    print(FF)
    # # FF = FF[0, :, 2]
    # y_pos = -0.6
    # xi = np.linspace(-design_region_width/2 - y_pos / np.tan(rot_angle),
    #                            design_region_width/2 - y_pos / np.tan(rot_angle),
    #                            design_region_width * resolution*5)
    # FF = [sim.get_field_point(mp.Ez, mp.Vector3(x=i, y=-half_total_height+y_pos)) for i in xi]
    #
    # print(FF)
    #
    # ideal = np.exp(1j * 2*pi * np.sin(rot_angle) * frequencies * xi)
    #
    # mode_overlap = np.abs(sum(np.conjugate(FF) * ideal)) ** 2 / (sum(np.abs(FF)**2) * sum(np.abs(ideal)**2))

    # return mode_overlap #
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
opt2 = mpa.OptimizationProblem(
    simulation=sim2,
    objective_functions=[J1],
    objective_arguments=ob_list2,
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

    f01, dJ_du1 = opt([mapping(reshaped_v[i, :], eta_i, cur_beta) for i in range(num_layers)]) # compute objective and gradient
    f02, dJ_du2 = opt2([mapping(reshaped_v[i, :], eta_i, cur_beta) for i in range(num_layers)])  # compute objective and gradient
    f0 = np.mean([f01, f02])
    dJ_du = average_list = [np.mean((arr1, arr2), axis=0) for arr1, arr2 in zip(dJ_du1, dJ_du2)]

    # f0 = np.mean([f01,f02])
    # shape of dJ_du [# degrees of freedom, # frequencies] or [# design regions, # degrees of freedom, # frequencies]

    if gradient.size > 0:
        if isinstance(dJ_du[0], list) or isinstance(dJ_du[0], np.ndarray):
            gradi = [tensor_jacobian_product(mapping, 0)(
                reshaped_v[i, :], eta_i, cur_beta, dJ_du[i]
            ) for i in range(num_layers)] # backprop
        else:
            gradi = tensor_jacobian_product(mapping, 0)(
                reshaped_v, eta_i, cur_beta, dJ_du) # backprop

        gradient[:] = np.reshape(gradi, [n])

    evaluation_history.append(np.real(f0)) # add objective function evaluation to list


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

num_samples = 10
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
        eps_parameters={'contour': False, 'alpha': 1, 'frequency': frequencies[0]},
        plot_sources_flag=False,
        plot_monitors_flag=False,
        plot_boundaries_flag=False,
        update_epsilon=True,  # required for the geometry to update dynamically
        nb=False  # True required if running in a Jupyter notebook
    )
    animateField = Animate2D(
        fields=mp.Ez,
        # realtime=True,
        eps_parameters={'contour': False, 'alpha': 1, 'frequency': frequencies[0]},
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
    # file_path = "x.npy"
    # with open(file_path, 'rb') as file:
    #     x = np.load(file)

    if True:
        for i in range(num_layers):
            x[Nx*i:Nx*(i+1)] = (npa.flipud(x[Nx*i:Nx*(i+1)]) + x[Nx*i:Nx*(i+1)]) / 2  # left-right symmetry
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
    cur_beta = 4 # 4
    beta_scale = 2 # 2
    num_betas = 7 # 6
    update_factor = 10 # 12
    totalIterations = num_betas * update_factor
    ftol = 1e-4 # 1e-5
    start = datetime.datetime.now()
    print("Opitimization started at " + str(start))
    # sendNotification("Opitimization started at " + str(start))
    for iters in range(num_betas):
        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(update_factor) # stop when 12 iterations are reached
        solver.set_ftol_rel(ftol)  # or when we converged
        x = solver.optimize(x)
        cur_beta = cur_beta * beta_scale
        estimatedSimulationTime = (datetime.datetime.now() - start) * num_betas / (iters + 1)
        print("Current iteration: {}".format(cur_iter[0]) + "; " + str(100 * cur_iter[0] / totalIterations) +
              "% completed ; eta at " + str(start + estimatedSimulationTime))
        # sendNotification("Opitimization " + str(100 * (iters + 1) / num_betas) + " % completed; eta at " +
        #                     str(start + estimatedSimulationTime))
        np.save("./" + scriptName + "/" + scriptName_i + "/x", x)
        plt.close()

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
    f0, dJ_du = opt([mapping(reshaped_x[i, :], eta_i, cur_beta // 2) for i in range(num_layers)], need_gradient=False)
    # frequencies = opt.frequencies

    if f0 > best_f0:
        best_f0 = f0
        best_design = x
        best_nr = sample_nr

    print("Objective_value = " + str(f0))

    # intensities = np.abs(opt.get_objective_arguments()[0][0, :, 2]) ** 2
    # print(opt.get_objective_arguments())

    # f0s[sample_nr, :] = intensities

    # Plot intensities
    # plt.figure()
    # plt.plot(1 / frequencies, intensities, "-o")
    # plt.grid(True)
    # plt.xlabel("Wavelength (microns)")
    # plt.ylabel("|Ez|^2 Intensities")
    # plt.savefig("./" + scriptName + "/" + scriptName_i + "/intensities.png")
    #
    # np.save("./" + scriptName + "/" + scriptName_i + "/v", x)
    #
    # animate.to_gif(fps=5, filename="./" + scriptName + "/" + scriptName_i + "/animation.gif")
    # animateField.to_gif(fps=5, filename="./" + scriptName + "/" + scriptName_i + "/animationField.gif")

    # Sy2 = 20
    # geometry.append(mp.Block(
    #     center=mp.Vector3(y=-(Sy2 / 2 + Sy / 2) / 2),
    #     size=mp.Vector3(x=Sx, y=(Sy2 / 2 - Sy / 2)),
    #     material=SiO2
    # ))
    space_below2 = 10  # includes PML
    Sy2 = half_total_height * 2 + space_below2
    cell_size = mp.Vector3(Sx, Sy2)
    source_center2 = [0, -(half_total_height - space_below2 / 2 + 0.4), 0]  # Source 1 µm below lens
    design_variables2 = [mp.MaterialGrid(mp.Vector3(Nx), SiO2, TiOx, grid_type="U_MEAN") for i in
                        range(num_layers)]  # SiO2
    design_regions2 = [mpa.DesignRegion(
        design_variables[i],
        volume=mp.Volume(
            center=mp.Vector3(y=-half_total_height + 0.5 * design_region_height[i] + sum(design_region_height[:i]) + i * spacing + space_below2 / 2),
            size=mp.Vector3(design_region_width, design_region_height[i], 0),
        ),
    ) for i in range(num_layers)]

    # Geometry: all is design region, no fixed parts
    geometry2 = [
        mp.Block(
            center=design_region.center, size=design_region.size, material=design_region.design_parameters
        )
        # mp.Block(center=design_region.center, size=design_region.size, material=design_variables, e1=mp.Vector3(x=-1))
        #
        # The commented lines above impose symmetry by overlapping design region with the same design variable. However,
        # currently there is an issue of doing that; instead, we use an alternative approach to impose symmetry.
        # See https://github.com/NanoComp/meep/issues/1984 and https://github.com/NanoComp/meep/issues/2093
        for design_region in design_regions2
    ]
    geometry2.append(mp.Block(
        center=mp.Vector3(y=(Sy2/2) - design_region_height[1] - (spacing/2)),
        size=mp.Vector3(x=Sx, y=spacing),
        material=SiO2
    ))

    #opt.update_design([mapping(reshaped_x[i, :], eta_i, cur_beta) for i in range(num_layers)])

    # Plot fields
    for freq in frequencies:
        src = mp.GaussianSource(frequency=freq, fwidth=fwidth)
        source3 = [mp.EigenModeSource(
            src,
            eig_band=1,
            direction=mp.NO_DIRECTION,
            eig_kpoint=kpoint,
            eig_parity=mp.ODD_Z,
            size=source_size,
            center=source_center2,
        )]
        source4 = [mp.EigenModeSource(
            src,
            eig_band=1,
            direction=mp.NO_DIRECTION,
            eig_kpoint=kpoint2,
            eig_parity=mp.ODD_Z,
            size=source_size,
            center=source_center2,
        )]
        opt.sim = mp.Simulation(
            cell_size=mp.Vector3(Sx, Sy2),
            boundary_layers=pml_layers,
            # k_point=kpoint,
            geometry=geometry2,
            sources=source3,
            default_material=Air,
            symmetries=[mp.Mirror(direction=mp.X)] if symmetry else None,
            resolution=resolution,
        )
        # src = mp.GaussianSource(frequency=freq, fwidth=fwidth)
        # source = [mp.EigenModeSource(
        #     src,
        #     eig_band=1,
        #     direction=mp.NO_DIRECTION,
        #     eig_kpoint=kpoint,
        #     eig_parity=mp.ODD_Z,
        #     size=source_size,
        #     center=source_center2,
        # )]
        # opt.sim.change_sources(source)

        opt.sim.run(until=200)
        plt.figure(figsize=(Sx, Sy2))
        opt.sim.plot2D(fields=mp.Ez)
        fileName = f"./" + scriptName + "/" + scriptName_i + "/fieldAtWavelength" + str(1/freq) + "Angle" + rot_angle + ".png"
        plt.savefig(fileName)

        opt.sim2 = mp.Simulation(
            cell_size=mp.Vector3(Sx, Sy2),
            boundary_layers=pml_layers,
            # k_point=kpoint,
            geometry=geometry2,
            sources=source4,
            default_material=Air,
            symmetries=[mp.Mirror(direction=mp.X)] if symmetry else None,
            resolution=resolution,
        )

        opt.sim2.run(until=200)
        plt.figure(figsize=(Sx, Sy2))
        opt.sim.plot2D(fields=mp.Ez)
        fileName = f"./" + scriptName + "/" + scriptName_i + "/fieldAtWavelength" + str(1 / freq) + "Angle" + rot_angle2 + ".png"
        plt.savefig(fileName)



    plt.close()

    estimatedSimulationTime = (datetime.datetime.now() - start0) * num_samples / (sample_nr + 1)
    # sendNotification("Opitimization " + str(int(10000 * (sample_nr + 1) / num_samples)/100) + " % completed; eta at " +
    #                  str(start0 + estimatedSimulationTime))

print(best_nr)
print(best_f0)

lb = np.min(f0s, axis=1)
ub = np.max(f0s, axis=1)
mean = np.mean(f0s, axis=1)

plt.figure()
plt.fill_between(np.arange(num_samples), ub, lb, alpha=0.3)
plt.plot(mean, "o-")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("FOM")
fileName = f"./" + scriptName + "/FOM.png"
plt.savefig(fileName)
plt.close()

with open("./" + scriptName + "/best_result.txt", 'w') as var_file:
    var_file.write("best_nr \t%d" % best_nr + "\n")
    var_file.write("best_f0 \t" + str(best_f0) + "\n")
    var_file.write("run_time \t" + str(datetime.datetime.now() - start0) + "\n")
    var_file.write("best_design \t" + str(best_design) + "\n")


# simulate intensities on best design
scriptName_i = "sample_" + str(best_nr)
opt.update_design([mapping(best_design[i, :], eta_i, cur_beta) for i in range(num_layers)])

Sy2 = 20
geometry.append(mp.Block(
    center=mp.Vector3(y=-(Sy2 / 2 + Sy / 2) / 2),
    size=mp.Vector3(x=Sx, y=(Sy2 / 2 - Sy / 2)),
    material=SiO2
))

# simulate intensities
sim = mp.Simulation(resolution=resolution,
                    cell_size=mp.Vector3(Sx, Sy2),
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    k_point=kpoint,
                    sources=source,
                    symmetries=[mp.Mirror(direction=mp.X)] if symmetry else None)

near_fields_focus = [sim.add_dft_fields([mp.Ez], freq, 0, 1, center=mp.Vector3(y=focal_point),
                                        size=mp.Vector3(x=design_region_width)) for freq in frequencies]
near_fields = [sim.add_dft_fields([mp.Ez], freq, 0, 1, center=mp.Vector3(),
                                  size=mp.Vector3(x=Sx, y=Sy2)) for freq in frequencies]
# near_fields_near = sim.add_dft_fields([mp.Ez], freq, 0, 1, center=mp.Vector3(y=half_total_height + 0.4),
#                                       size=mp.Vector3(x=design_region_width))

sim.run(until_after_sources=100)

focussed_field = [sim.get_dft_array(field, mp.Ez, 0) for field in near_fields_focus]
scattered_field = [sim.get_dft_array(field, mp.Ez, 0) for field in near_fields]
# near_field = sim.get_dft_array(near_fields, mp.Ez, 0)

focussed_amplitude = [np.abs(field) ** 2 for field in focussed_field]
scattered_amplitude = [np.abs(field) ** 2 for field in scattered_field]
# print(scattered_amplitude)
# print(sim.get_array_metadata(dft_cell=near_fields))
# print(np.shape(sim.get_array_metadata(dft_cell=near_fields)))
[xi, yi, zi, wi] = sim.get_array_metadata(dft_cell=near_fields_focus[1])
[xj, yj, zj, wj] = sim.get_array_metadata(dft_cell=near_fields[1])

for i in range(nf):
    # plot colormesh
    plt.figure(dpi=150)
    plt.pcolormesh(xj, yj, np.rot90(np.rot90(np.rot90(scattered_amplitude[i]))), cmap='inferno', shading='gouraud',
                   vmin=0,
                   vmax=scattered_amplitude[i].max())
    plt.gca().set_aspect('equal')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    # ensure that the height of the colobar matches that of the plot
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    plt.tight_layout()
    fileName = f"./" + scriptName + "/" + scriptName_i + "/intensityMapAtWavelength" + str(1 / frequencies[i]) + ".png"
    plt.savefig(fileName)

    # plot intensity around focal point
    plt.figure()
    plt.plot(xi, focussed_amplitude[i], 'bo-')
    plt.xlabel("x (μm)")
    plt.ylabel("field amplitude")
    fileName = f"./" + scriptName + "/" + scriptName_i + "/intensityOverLineAtWavelength" + str(
        1 / frequencies[i]) + ".png"
    plt.savefig(fileName)

np.save("./" + scriptName + "/" + scriptName_i + "/intensity_at_focus", focussed_amplitude)

# sendNotification("Simulation finished; best FOM: " + str(best_f0))
# sendPhoto(fileName)
