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

scriptName = "metalens_empty_contin"

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
# pml_layers = [mp.PML(pml_size, direction=d) for d in [mp.Y]] # only PML in y-direction

fcen = 1 / 1.55 # Middle frequency of source
width = 0.2 # Relative width of frequency
fwidth = width * fcen # Absolute width of frequency
source_center = [0, -(design_region_height / 2 + 1.5), 0] # Source 1.5 µm below lens
source_size = mp.Vector3(Sx, 0, 0) # Source covers width of lens # design_region_width instead of Sx
src = mp.ContinuousSource(frequency=fcen) # Gaussian source
source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]

# Amount of variables
Nx = int(design_region_resolution * design_region_width)
Ny = 1

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
    filtered_field = conic_filter2( # remain minimum feature size
        x,
        filter_radius,
        design_region_width,
        design_region_height,
        Nx,
        Ny
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
# kpoint = mp.Vector3(x=Sx) # set periodic boundary conditions in x-direction
sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    # k_point= kpoint,
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

# Upload x
# file_path = filedialog.askopenfilename(filetypes=[("Numpy Files", "*.npy")], initialdir='\\wsl.localhost\\Ubuntu-22.04\\home\\willem\\PO_Nano_code')
# file_path = "x.npy"
# with open(file_path, 'rb') as f:
#     x = np.load(f)
x = np.zeros(Nx,)

# insert dummy parameter bounds and variable
x = np.insert(x, 0, -1)  # our initial guess for the worst error

cur_beta = 256 # 4

# Check intensities in optimal design
# f0, dJ_du = opt([mapping(x[1:], eta_i, cur_beta // 2)], need_gradient=False)
# frequencies = opt.frequencies
#
# intensities = np.abs(opt.get_objective_arguments()[0][0, :, 2]) ** 2
#
# # Plot intensities
# plt.figure()
# plt.plot(1 / frequencies, intensities, "-o")
# plt.grid(True)
# plt.xlabel("Wavelength (microns)")
# plt.ylabel("|Ez|^2 Intensities")
# fileName = f"./" + scriptName + "_img/intensities.png"
# plt.savefig(fileName)

# Plot fields
for freq in frequencies:
    opt.sim = mp.Simulation(
        cell_size=mp.Vector3(Sx, 40),
        boundary_layers=pml_layers,
        # k_point=kpoint,
        geometry=geometry,
        sources=source,
        default_material=Air,
        resolution=resolution,
    )
    src = mp.ContinuousSource(frequency=freq)
    source = [mp.Source(src, component=mp.Ez, size=source_size, center=source_center)]
    opt.sim.change_sources(source)

    opt.sim.run(until=200)
    plt.figure(figsize=(10, 20))
    opt.sim.plot2D(fields=mp.Ez)
    fileName = f"./" + scriptName + "_img/fieldAtWavelength" + str(1/freq) + ".png"
    plt.savefig(fileName)

np.save("./" + scriptName + "_img/x.npy", x)

plt.close()


