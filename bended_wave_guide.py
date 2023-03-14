import meep as mp
import numpy as np
import matplotlib.pyplot as plt

print("----------- Running Linear wave guide.py -----------")
print("----------- Initializing geometry -----------")

# Size of cell in x-, y- and z-direction (in µm)
cell = mp.Vector3(16,16,0)

# Any place not specified has epsilon = 1
# Create infinite slab of thickness 1 (in y-direction) with epsilon = 12, centered at (0,0)
geometry = [mp.Block(mp.Vector3(10,1,mp.inf),
                     center=mp.Vector3(-3, 2),
                     material=mp.Medium(epsilon=12)),
            mp.Block(mp.Vector3(1, 10.5, mp.inf),
                     center=mp.Vector3(2, -2.75),
                     material=mp.Medium(epsilon=12))]

# wavelength in µm (2*sqrt(11)); width: turn on source slowly over 20 time units
sources = [mp.Source(mp.ContinuousSource(wavelength=2*(11**0.5), width=20),
                     component=mp.Ez, # transversal polarization
                     center=mp.Vector3(-7, 2), # Located on the left side (1 cm from edge)
                     size=mp.Vector3(0,1))] # Line source, instead of point source

# Boundary conditions (perfecly matched layers = absorbing of thickness 1 (inside cell!!!))
pml_layers = [mp.PML(1.0)]

# Pixels per µm. 10 gives 20 pixels per wavelength in epsilon = 12, you need atleast 8
resolution = 10

# Simulation object with
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

print("----------- Starting simulation -----------")

sim.run(until=200)

print("----------- Simulation completed -----------")
# Save data
print("----------- Saving data -----------")

eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
np.save("epsilon", eps_data)
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
np.save("Ez", ez_data)

# Plot data
print("----------- Plotting data -----------")

plt.figure()
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.axis('off')
plt.savefig("Epsilon.png")

plt.figure()
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
plt.axis('off')
plt.savefig("Ez.png")
plt.show()
