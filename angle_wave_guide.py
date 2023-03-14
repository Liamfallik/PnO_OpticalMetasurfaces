import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from math import *

print("----------- Running angle_wave_guide.py -----------")
print("----------- Initializing geometry -----------")

# Size of cell in x-, y- and z-direction (in µm)
box = 16
cell = mp.Vector3(box,box,0)

epsi = 12
length = 10
distance = 6
t = 1
angle = float(input("Angle (between 1° and 89°): "))*(-pi/180)
part2 = mp.Vector3((box-distance+t/2)/abs(sin(angle)), t/cos(angle), mp.inf)
part2.rotate(mp.Vector3(0, 0, 1), angle)

# Any place not specified has epsilon = 1
# Create infinite slab of thickness 1 (in y-direction) with epsilon = 12, centered at (0,0)
geometry = [mp.Block(mp.Vector3(length,t,mp.inf),
                     center=mp.Vector3(-(box-length)/2, box/2-distance),
                     material=mp.Medium(epsilon=epsi)),
            mp.Block(part2,
                     mp.Vector3(1, 0, 0).rotate(mp.Vector3(0, 0, 1), angle),
                     center=mp.Vector3(-box/2+length + (box-distance+t/2)/abs(2*tan(angle)), - distance/2 - t/4),
                     material=mp.Medium(epsilon=epsi))]

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
