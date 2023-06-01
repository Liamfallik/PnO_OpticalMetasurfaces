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
import atexit

scriptName = "3d_3fp_cont"
num_layers = 4
start_from_direct = False
symmetry = False # NOT IMPLEMENTED YET
#mp.divide_parallel_processes(16)


def sendNotification(message):
    token = "5930732844:AAEiVTJ9ppOM3RJj3gSxImKxk7vaRTAQu-0"
    myuserid = 6099565081
    method = '/sendMessage'
    url = f"https://api.telegram.org/bot{token}"
    params = {"chat_id": myuserid, "text": message}
    try:
        r = requests.get(url + method, params=params)
    except:
        print("No internet connection: couldn't send notification...")

def sendPhoto(image_path):
    token = "5930732844:AAEiVTJ9ppOM3RJj3gSxImKxk7vaRTAQu-0"
    myuserid = 6099565081
    data = {"chat_id": myuserid}
    url = f"https://api.telegram.org/bot{token}" + "/sendPhoto"
    with open(image_path, "rb") as image_file:
        try:
            ret = requests.post(url, data=data, files={"photo": image_file})
            return ret.json()
        except:
            print("No internet connection: couldn't send notification...")

def aterror():
    if mp.am_really_master():
        sendNotification("Script stopped")
    
atexit.register(aterror)

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
    x = np.array(x)
    x2 = np.reshape(x, [Nx, Ny])
    halfx = int(Nx/2)
    halfy = int(Ny/2)
    x2[0:halfx, 0:halfy] = np.fliplr(x2[0:halfx, halfy:Ny])
    x2[halfx:Nx, 0:halfy] = np.flipud(np.fliplr(x2[0:halfx, halfy:Ny]))
    x2[halfx:Nx, halfy:Ny] = np.flipud(x2[0:halfx, halfy:Ny])
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
design_region_width = 3.4
design_region_height = [0.4, 0.365, 0.298, 0.186] # for 3 fp
#design_region_height = [0.487, 0.299, 0.230, 0.138]
#design_region_height = [0.25]
spacing = 0
half_total_height = sum(design_region_height) / 2 + (num_layers - 1) * spacing / 2

# Boundary conditions
pml_size = 0.3
resolution = 50 #50

# System size
far_field_distance = 0.3
source_distance = 0.3
Sx = 2 * pml_size + design_region_width
Sz = 2 * pml_size + half_total_height * 2 + far_field_distance + source_distance + 0.4
cell_size = mp.Vector3(Sx, Sx, Sz)

# Frequencies
f_red = 1 / 0.65
f_blue = 1 / 0.47
f_green = 1/0.55
frequencies = [f_red, f_green, f_blue]
#frequencies = [f_red, f_blue]

# Feature size constraints
minimum_length = 0.04  # minimum length scale (microns)
eta_i = (
    0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
)
eta_e = 0.75  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution)

# Boundary conditions
pml_layers = [mp.PML(pml_size)]

fwidth = 0.2  # Relative width of frequency
source_pos = -(half_total_height + source_distance)
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
    #projected_field = symmetrize(projected_field, Nx, Ny)

    # interpolate to actual materials
    return projected_field.flatten()


# Lens geometry
geometry = [
	mp.Block(
        center=design_region.center, size=design_region.size, material=design_region.design_parameters
    )
for design_region in design_regions]

""""
geometry.append(mp.Block(
        center = mp.Vector3(z=-(half_total_height + Sz/2) / 2),
        size=mp.Vector3(x = Sx, y = Sx, z = (Sz/2 - half_total_height)),
        material=SiO2
    ))
"""

# Set-up simulation object
kpoint = mp.Vector3()
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air,
    #symmetries=[mp.Mirror(direction=mp.X, phase=-1), mp.Mirror(direction=mp.Y)],
    # symmetries=[mp.Mirror(direction=mp.Y)] if symmetry else None,
    #k_point=kpoint,
    resolution=resolution,
    #eps_averaging=False,
)

# Focus point, 6 uM beyond centre of lens
focal_point = 3
#far_x = [mp.Vector3(0, 0, focal_point), mp.Vector3(0, 3, focal_point), mp.Vector3(3, 0, focal_point), mp.Vector3(3, 3, focal_point)] # Blue, Green, Green, Red
far_x = [mp.Vector3(-1.5, -1.5, focal_point), mp.Vector3(-1.5, 1.5, focal_point), mp.Vector3(1.5, -1.5, focal_point), mp.Vector3(1.5, 1.5, focal_point)] # Blue, Green, Green, Red
#far_x = [mp.Vector3(-1.5, -1.5, focal_point), mp.Vector3(1.5, 1.5, focal_point)] # Blue, Green, Green, Red
NearRegions = [  # region from which fields at focus point will be calculated
    mp.Near2FarRegion(
        center=mp.Vector3(0, 0, half_total_height + far_field_distance),  # 0.3 um above lens
        size=mp.Vector3(design_region_width, design_region_width, 0),  # spans design region
        weight=+1,  # field contribution is positive (real)
    )
]

FarFields = mpa.Near2FarFields(sim, NearRegions, far_x)  # Far-field object
ob_list = [FarFields]


def J1(FF):
    """
    Returns the minimum between the value for |Ex|2 for both frequencies at their respective focal points

    Parameters
    ----------
    FF - a list of Near2FarFields objects, representing a 3d matrix with:
        [0] the points where the fields are calculated
        [1] the different frequencies at which they are calculated
        [2] the values for Ex, Ey, Ez, Hx, Hy, Hz

    Returns
    -------
    The value of the objective function to maximize
    """
    # Frequencies: [R, G, B]
    # Points: [B, G, G, R]
    
    print(str(np.abs(FF[0, 2, 0]) ** 2) + ", " +  str(np.abs(FF[1, 1, 0]) ** 2) + ", " +  str(np.abs(FF[2, 1, 0]) ** 2) + ", " +  str(np.abs(FF[3, 0, 0]) ** 2))
    return min([(np.abs(FF[0, 2, 0]) ** 2), (np.abs(FF[1, 1, 0]) ** 2), (np.abs(FF[2, 1, 0]) ** 2),(np.abs(FF[3, 0, 0]) ** 2)])
    
    
    #return np.abs(FF[0, 2, 0]) ** 2
    #return min([(np.abs(FF[0, 1, 0]) ** 2),(np.abs(FF[1, 0, 0]) ** 2)])


# Optimization object
opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J1],
    objective_arguments=ob_list,
    design_regions=design_regions,
    frequencies=frequencies,
    maximum_run_time=200, # 500
)

# Gradient
evaluation_history = []  # Keep track of objective function evaluations
cur_iter = [0]  # Iteration


def f(v, gradient, cur_beta):
    print(min(v))
    print(max(v))
    print(cur_beta)
    if cur_iter[0] != 0:
        estimatedSimulationTime = (datetime.datetime.now() - start) * totalIterations / cur_iter[0]
        print("Current iteration: {}".format(cur_iter[0]) + "; " + str(100 * cur_iter[0] / totalIterations) +
              "% completed ; eta at " + str(start + estimatedSimulationTime))
    else:
        print("Current iteration: {}".format(cur_iter[0]))
    cur_iter[0] += 1

    reshaped_v = np.reshape(v, [num_layers, Nx*Ny])

    f0, dJ_du = opt(
        [mapping(reshaped_v[i, :], eta_i, cur_beta) for i in range(num_layers)])  # compute objective and gradient
    # shape of dJ_du [# degrees of freedom, # frequencies] or [# design regions, # degrees of freedom, # frequencies]

    if gradient.size > 0:
        if isinstance(dJ_du[0], list) or isinstance(dJ_du[0], np.ndarray):
            gradi = [tensor_jacobian_product(mapping, 0)(
                reshaped_v[i, :], eta_i, cur_beta, np.sum(dJ_du[i], axis=1)
            ) for i in range(num_layers)]  # backprop
        else:
            gradi = tensor_jacobian_product(mapping, 0)(
                reshaped_v, eta_i, cur_beta, dJ_du)  # backprop
        print(np.asarray(gradi).shape)
        gradient[:] = np.reshape(gradi, [n])

    evaluation_history.append(f0)  # add objective function evaluation to list

    opt.update_design([mapping(reshaped_v[i, :], eta_i, cur_beta) for i in range(num_layers)])
    print(gradient)
    print("f0: " + str(f0))
    if cur_iter[0]%5 == 0:
        np.save("./" + scriptName + "/x_" + str(cur_iter[0]), x)
    return f0

    if mp.am_really_master() and cur_iter[0] % 20 == 0:
        np.save("./" + scriptName + "/x_" + str(cur_iter[0]), x)
        for i in range(num_layers):
            plt.figure()  # Plot current design
            ax = plt.gca()
            opt.plot2D(
                False,
                ax=ax,
                plot_sources_flag=False,
                plot_monitors_flag=False,
                plot_boundaries_flag=False,
                output_plane=output_planes[i]
            )
            circ = Circle((2, 2), minimum_length / 2)
            ax.add_patch(circ)
            ax.axis("off")
            plt.savefig("./" + scriptName + "/layer" + str(i) + "_img_{" + str(cur_iter[0]) + "}.png")

    plt.close()

    return f0

# Method of moving  asymptotes
algorithm = nlopt.LD_MMA
n = Nx * Ny * num_layers # number of parameters

seed = 35 # make sure starting conditions are random, but always the same. Change seed to change starting conditions
np.random.seed(seed)
# Initial guess
#reshaped_x = np.zeros([num_layers, Nx, Ny])
#x = np.random.rand(n)
x = np.ones(n) * 0.5

start_beta = 256 # 4
reshaped_x = np.reshape(x, [num_layers, Nx*Nx])
mapped_x = [mapping(reshaped_x[i, :], eta_i, start_beta) for i in range(num_layers)]
opt.update_design(mapped_x)
#print(mapped_x[0].shape)


x = np.load("3d_3fp/x_final.npy")
reshaped_x = np.reshape(x, [num_layers, Nx*Nx])
opt.update_design([mapping(reshaped_x[i, :], eta_i, start_beta) for i in range(num_layers)])
print(min(x))
print(max(x))


# lower and upper bounds
lb = np.zeros(n)
ub = np.ones(n)

if mp.am_really_master():
	print("plot")
	output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(Sx, 0, 10))
	plt.figure()
	opt.plot2D(output_plane=output_plane, plot_monitors_flag=True)
	plt.savefig(scriptName + "/design_XZ.png")

	output_planes = [mp.Volume(center=design_region.center, size=mp.Vector3(Sx, Sx, 0)) for design_region in design_regions]
	for i in range(num_layers):
		plt.figure()
		opt.plot2D(output_plane=output_planes[i])
		plt.savefig(scriptName + "/design_layer" + str(i) + ".png")

cur_beta = start_beta
beta_scale = 4
num_betas = 1 #6
update_factor = 80 #20
totalIterations = num_betas * update_factor
ftol = 0.00001
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
    var_file.write("wavelengths \t" + str([1/frequency for frequency in frequencies]) + "\n")
    var_file.write("fwidth \t" + str(fwidth) + "\n")
    var_file.write("focal_point \t" + str(focal_point) + "\n")
    var_file.write("start_beta \t" + str(start_beta) + "\n")
    var_file.write("beta_scale \t" + str(beta_scale) + "\n")
    var_file.write("num_betas \t" + str(num_betas) + "\n")
    var_file.write("update_factor \t" + str(update_factor) + "\n")
    var_file.write("ftol \t" + str(ftol) + "\n")
    var_file.write("seed \t%d" % seed + "\n")

print("Opitimization started at " + str(start))
if mp.am_really_master():
    sendNotification("Opitimization started at " + str(start))
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
    if mp.am_really_master():
        sendNotification("Opitimization " + str(100 * (iters + 1) / num_betas) + " % completed; eta at " +
                         str(start + estimatedSimulationTime))
    np.save("./" + scriptName + "/x_" + str(cur_iter[0]), x)

cur_beta = cur_beta / beta_scale
np.save("./" + scriptName + "/x_final", x)
# x = np.zeros(n) # REMOVE THIS
# Check intensities in optimal design
reshaped_x = np.reshape(x, [num_layers, Nx*Ny])
mapped_x = [mapping(reshaped_x[i, :], eta_i, 2000) for i in range(num_layers)]

f0, dJ_du = opt(mapped_x, need_gradient=False)
frequencies = opt.frequencies

intensities = np.abs(opt.get_objective_arguments()[0][0, :, 0])

# Plot intensities
plt.figure()
plt.plot([1 / fr for fr in frequencies], intensities, "-o")
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
opt.update_design(mapped_x)
for i in range(num_layers):
    plt.figure()
    opt.plot2D(output_plane=output_planes[i])
    plt.savefig(scriptName + "/final_design_layer" + str(i) + ".png")


with open("./" + scriptName + "/best_result.txt", 'w') as var_file:
    var_file.write("f0 \t" + str(f0) + "\n")
    var_file.write("best_design \t" + str(x) + "\n")
"""
# Field plot
Sz2 = 10
geometry.append(mp.Block(
    center=mp.Vector3(z=-(Sz2 / 2 + Sz / 2) / 2),
    size=mp.Vector3(x=Sx, y=Sx, z=(Sz2 / 2 - Sz / 2)),
    material=SiO2
))

# Plot fields
for freq in frequencies:
	sim = mp.Simulation(
		cell_size=mp.Vector3(Sx, Sx, Sz2),
		boundary_layers=pml_layers,
		# k_point=kpoint,
		geometry=geometry,
		sources=source,
		default_material=Air,
		# symmetries=[mp.Mirror(direction=mp.X), mp.Mirror(direction=mp.Y)] if symmetry else None,
		# symmetries=[mp.Mirror(direction=mp.Y)] if symmetry else None,
		resolution=resolution,
	)

	near_fields_focus = sim.add_dft_fields([mp.Ex], freq, 0, 1, center=mp.Vector3(z=focal_point),
		                                       size=mp.Vector3(x=design_region_width, y=design_region_width))
	near_fields_focus_line = sim.add_dft_fields([mp.Ex], freq, 0, 1, center=mp.Vector3(z=focal_point),
		                                       size=mp.Vector3(x=design_region_width))
	near_fields_before = sim.add_dft_fields([mp.Ex], freq, 0, 1,
		                                        center=mp.Vector3(z=-(-half_total_height + source_pos) / 2),
		                                        size=mp.Vector3(x=design_region_width, y=design_region_width))
	near_fields = sim.add_dft_fields([mp.Ex], freq, 0, 1, center=mp.Vector3(),
		                                size=mp.Vector3(x=Sx, z=Sz2))



	sim.run(until=200)

	if mp.am_really_master():
		plt.figure()#figsize=(Sx, Sz2))
		output_plane = mp.Volume(center=mp.Vector3(), size=mp.Vector3(Sx, 0, Sz2))
		# opt.sim.plot2D(
		#             False,
		#             fields=mp.Ex,
		#             output_plane=output_plane)
		ax = plt.gca()


		sim.plot2D(
		    # False,
		    ax=ax,
		    # plot_sources_flag=False,
		    plot_monitors_flag=False,
		    # plot_boundaries_flag=False,
		    output_plane=output_plane,
		    fields=mp.Ex
		)
		fileName = "./" + scriptName + "/field.png"
		plt.savefig(fileName)

		focussed_field = sim.get_dft_array(near_fields_focus, mp.Ex, 0)
		focussed_field_line = sim.get_dft_array(near_fields_focus_line, mp.Ex, 0)
		before_field = sim.get_dft_array(near_fields_before, mp.Ex, 0)
		scattered_field = sim.get_dft_array(near_fields, mp.Ex, 0)
		# near_field = sim.get_dft_array(near_fields, mp.Ex, 0)

		print(focussed_field_line)
		focussed_amplitude = np.abs(focussed_field) ** 2
		focussed_amplitude_line = np.abs(focussed_field_line) ** 2
		scattered_amplitude = np.abs(scattered_field) ** 2
		before_amplitude = np.abs(before_field) ** 2

		[xi, yi, zi, wi] = sim.get_array_metadata(dft_cell=near_fields_focus)
		[xk, yk, zk, wk] = sim.get_array_metadata(dft_cell=near_fields_focus_line)
		[xj, yj, zj, wj] = sim.get_array_metadata(dft_cell=near_fields)

		# plot intensity XZ
		plt.figure(dpi=150)
		plt.pcolormesh(xj, zj, np.rot90(np.rot90(np.rot90(scattered_amplitude))), cmap='inferno', shading='gouraud',
		               vmin=0,
		               vmax=scattered_amplitude.max())
		plt.gca().set_aspect('equal')
		plt.xlabel('x (μm)')
		plt.ylabel('z (μm)')

		# ensure that the height of the colobar matches that of the plot
		from mpl_toolkits.axes_grid1 import make_axes_locatable

		divider = make_axes_locatable(plt.gca())
		cax = divider.append_axes("right", size="5%", pad=0.05)
		plt.colorbar(cax=cax)
		plt.tight_layout()
		fileName = f"./" + scriptName + "/intensityMapXZ" + ".png"
		plt.savefig(fileName)

		# plot intensity XY focal
		plt.figure(dpi=150)
		plt.pcolormesh(xi, yi, np.rot90(np.rot90(np.rot90(focussed_amplitude))), cmap='inferno', shading='gouraud',
		               vmin=0,
		               vmax=focussed_amplitude.max())
		plt.gca().set_aspect('equal')
		plt.xlabel('x (μm)')
		plt.ylabel('y (μm)')

		# ensure that the height of the colobar matches that of the plot
		from mpl_toolkits.axes_grid1 import make_axes_locatable

		divider = make_axes_locatable(plt.gca())
		cax = divider.append_axes("right", size="5%", pad=0.05)
		plt.colorbar(cax=cax)
		plt.tight_layout()
		fileName = f"./" + scriptName + "/intensityFocusMapXY" + ".png"
		plt.savefig(fileName)

		# plot intensity XY before
		plt.figure(dpi=150)
		plt.pcolormesh(xi, yi, np.rot90(np.rot90(np.rot90(before_amplitude))), cmap='inferno', shading='gouraud',
		               vmin=0,
		               vmax=before_amplitude.max())
		plt.gca().set_aspect('equal')
		plt.xlabel('x (μm)')
		plt.ylabel('y (μm)')

		# ensure that the height of the colobar matches that of the plot
		from mpl_toolkits.axes_grid1 import make_axes_locatable

		divider = make_axes_locatable(plt.gca())
		cax = divider.append_axes("right", size="5%", pad=0.05)
		plt.colorbar(cax=cax)
		plt.tight_layout()
		fileName = f"./" + scriptName + "/intensityMapXY" + ".png"
		plt.savefig(fileName)

		# plot intensity around focal point
		plt.figure()
		plt.plot(xk, focussed_amplitude_line, 'bo-')
		plt.xlabel("x (μm)")
		plt.ylabel("field amplitude")
		fileName = f"./" + scriptName + "/intensityOverLine_atFocalPoint.png"
		plt.savefig(fileName)
		sendPhoto(fileName)
		# print(focussed_amplitude_line)
		efficiency = focussing_efficiency(focussed_amplitude_line, focussed_amplitude_line)
		FWHM = get_FWHM(focussed_amplitude_line, xk)


		np.save("./" + scriptName + "/intensity_at_focus",
		        focussed_amplitude_line)

with open("./" + scriptName + "/best_result.txt", 'a') as var_file:
    var_file.write("focussing_efficiency \t" + str(efficiency) + "\n")
    var_file.write("FWHM \t" + str(FWHM) + "\n")
    var_file.write("run_time \t" + str(datetime.datetime.now() - start) + "\n")
"""
sendNotification("Simulation finished")
