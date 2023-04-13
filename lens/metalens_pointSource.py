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

scriptName = "pointSource_4l2fp_blue"
symmetry = False # Impose symmetry around x = 0 line

def sendNotification(message):
	token = "5930732844:AAEiVTJ9ppOM3RJj3gSxImKxk7vaRTAQu-0"
	myuserid = 6099565081
	method = '/sendMessage'
	url = "https://api.telegram.org/bot{0}".format(token) + "/sendMessage"
	params = {"chat_id": myuserid, "text": message}
	try:
		print(message)
		response = requests.post(url, json={'chat_id': myuserid, 'text': scriptName + ":\n" + message})
		print(response.text)
	except Exception as e:
		print(e)


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
num_layers = 4 # amount of layers
design_region_width = 10 # width of layer
design_region_height = [0.487, 0.299, 0.230, 0.138] # height of layer
spacing = 0 # spacing between layers
half_total_height = sum(design_region_height) / 2 + (num_layers - 1) * spacing / 2
empty_space = 0 # free space in simulation left and right of layer

# Boundary conditions
pml_size = 1.0 # thickness of absorbing boundary layer

resolution = 55 # 50 --> amount of grid points per um; needs to be > 49 for TiOx and 0.55 um

# System size
Sx = 2 * pml_size + design_region_width + 2 * empty_space
# Sx = design_region_width + 2 * empty_space
Sy = 2 * pml_size + half_total_height * 2 + 2
cell_size = mp.Vector3(Sx, Sy)

# Frequencies
# nf = 3 # Amount of frequencies studied
# frequencies = 1./np.linspace(0.55, 0.65, nf)
f_red = 1 / 0.65
f_blue = 1 / 0.47
frequencies = [f_red, f_blue]

# Feature size constraints
minimum_length = 0.09  # minimum length scale (microns)
eta_i = (
    0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
)
eta_e = 0.65  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
design_region_resolution = int(resolution) # = int(resolution)

# Boundary conditions
pml_layers = [mp.PML(pml_size)]

# Source
# fcen = frequencies[1]
fwidth = 0.03 # 0.2
source_center = [2, 6, 0]
source_size = mp.Vector3(0, 0, 0) # Source covers width of lens
# src = mp.GaussianSource(frequency=fcen, fwidth=fwidth) # Gaussian source
# source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
srcs = [mp.GaussianSource(frequency=fcen, fwidth=fwidth) for fcen in frequencies] # Gaussian source
source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center) for src in srcs]

# Amount of variables
Nx = int(design_region_resolution * design_region_width)
Ny = 1 # int(design_region_resolution * design_region_height)
print(Nx)


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

    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta) # Make binary

    if symmetry:
        projected_field = (
            npa.flipud(projected_field) + projected_field
        ) / 2  # left-right symmetry

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
# geometry.append(mp.Block(
#         center = mp.Vector3(y=-(half_total_height + Sy/2) / 2),
#         size=mp.Vector3(x = Sx, y = (Sy/2 - half_total_height)),
#         material=SiO2
#     ))

block_height = 8

# Geometry: all is design region, no fixed parts
# geometry = [
#     mp.Block(
#         center=design_region.center, size=design_region.size, material=design_variables
#     ) for design_region in design_regions]
    
geometry.append(mp.Block(center=mp.Vector3(y=(block_height + sum(design_region_height))/2), size=mp.Vector3(Sx, block_height), material=SiO2))

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

# Focus point, 7.5 um beyond centre of lens
focal_point = 4
far_x = [mp.Vector3(-2, focal_point, 0), mp.Vector3(2, focal_point, 0)]
NearRegions = [ # region from which fields at focus point will be calculated
    mp.Near2FarRegion(
        center=mp.Vector3(0, half_total_height + 0.4), # 0.4 um above lens
        size=mp.Vector3(design_region_width + 2*empty_space, 0), # spans design region
        weight=+1, # field contribution is positive (real)
    )
]
FarFields = mpa.Near2FarFields(sim, NearRegions, far_x) # Far-field object
ob_list = [FarFields]


def J1(FF):
    return min((np.abs(FF[0, 0, 2]) ** 2), (np.abs(FF[1, 1, 2]) ** 2))

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
    plt.savefig("./" + scriptName + "/" + scriptName_i + "/img_" + str(cur_iter[0]) + ".png")

    # Efield = opt.sim.get_epsilon()
    # plt.figure()
    # plt.plot(Efield ** 2)

    plt.close()

    return np.real(f0)


# Initial guess
seed = 1425 # make sure starting conditions are random, but always the same. Change seed to change starting conditions
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
    #var_file.write("wavelengths \t" + str(1/frequencies) + "\n")
    var_file.write("fwidth \t" + str(fwidth) + "\n")
    var_file.write("focal_point \t" + str(focal_point) + "\n")
    var_file.write("seed \t%d" % seed + "\n")

num_samples = 1
# store best objective value
best_f0 = 0
best_design = None
best_nr = None
f0s = np.zeros([num_samples, len(frequencies)])

for sample_nr in range(num_samples):
	opt = mpa.OptimizationProblem(
		simulation=sim,
		objective_functions=[J1], 
		objective_arguments=ob_list, 
		design_regions=design_regions,
		frequencies=frequencies, 
		maximum_run_time=2000)

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

	x = np.random.rand(n) #* 0.6
	if symmetry:
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
	print(reshaped_x)
	#mapped_x = [mapping(reshaped_x[i, :], eta_i, 4) for i in range(num_layers)]
	#print(mapped_x)
	opt.update_design(reshaped_x)
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

    
	with open("4l2fp_willem1/sample_2/x.npy", 'rb') as f:
	    x = np.load(f)

	print(len(x))
	cur_beta = 256
	#opt.update_design([x[0:549], 1, 1])
#, x[550:1099], x[1100:1649], x[1650:2199]])


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
	frequencies = opt.frequencies

	if f0 > best_f0:
		best_f0 = f0
		best_design = x
		best_nr = sample_nr

	print("Objective_value = " + str(f0))

	intensities = [np.abs(opt.get_objective_arguments()[0][0, 0, 2] ** 2), np.abs(opt.get_objective_arguments()[0][1, 1, 2] ** 2)]
	print(opt.get_objective_arguments())

	f0s[sample_nr, :] = intensities

	# Plot intensities
	plt.figure()
	plt.plot([1/freqq for freqq in frequencies], intensities, "-o")
	plt.grid(True)
	plt.xlabel("Wavelength (microns)")
	plt.ylabel("|Ez|^2 Intensities")
	plt.savefig("./" + scriptName + "/" + scriptName_i + "/intensities.png")

	np.save("./" + scriptName + "/" + scriptName_i + "/v", x)

	animate.to_gif(fps=5, filename="./" + scriptName + "/" + scriptName_i + "/animation.gif")
	animateField.to_gif(fps=5, filename="./" + scriptName + "/" + scriptName_i + "/animationField.gif")


    # Plot fields
	for freq in frequencies:
		Sy2 = 20
		geometry.append(mp.Block(
			center=mp.Vector3(y=(Sy2 / 2 + Sy / 2) / 2),
			size=mp.Vector3(x=Sx, y=(Sy2 / 2 - Sy / 2)),
			material=SiO2
		))
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
		fileName = "./" + scriptName + "/" + scriptName_i + "/fieldAtWavelength" + str(1/freq) + ".png"
		plt.savefig(fileName)
		try:
			Efield = opt.get_efield_z()
			print(Efield)
			plt.figure()
			plt.imshow(np.abs(Efield)**2, interpolation="nearest", origin="upper")
			plt.colorbar()
			fileName = "./" + scriptName + "/" + scriptName_i + "/intensityAtWavelength" + str(1 / freq) + ".png"
			plt.savefig(fileName)
		except Exception as e:
			print("Plotting intensity failed, needs updated meep files: " + str(e))


	plt.close()

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
fileName = "./" + scriptName + "/FOM.png"
plt.savefig(fileName)
plt.close()

with open("./" + scriptName + "/best_result.txt", 'w') as var_file:
    var_file.write("best_nr \t%d" % best_nr + "\n")
    var_file.write("best_f0 \t" + str(best_f0) + "\n")
    var_file.write("best_design \t" + str(best_design) + "\n")

sendNotification("Optimization *" + scriptName + "* finished.")
