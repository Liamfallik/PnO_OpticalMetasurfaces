import numpy as np
import random
from matplotlib import pyplot as plt

"""
Simulated annealing optimization of the thicknesses of a lens.
"""

frequencies = [0.47, 0.65] # frequencies that you want to optimise independently
num_layers = 4

max_thickness = 0.5 # maximum thickness of a single layer
deltan = 2.7 - 1.45 # difference in refractive index

step_size = 0.1

def phase(freqs, thickness, deltan):
    """
    Returns the phase shift of a light wave with a certain frequency freq induced by a path through a certain thickness
    compared between materials with a difference in refractive index deltan.
    """
    return [thickness * deltan / freq % 1 for freq in freqs]


def get_all_phases(freqs, layers, deltan):
    """
    All possible phases shifts of a light wave with a certain frequency freq induced by a path through all possible
    thicknesses that are possible to all possible combinations of the layers
    compared between materials with a difference in refractive index deltan.
    """
    num_layers = max(np.shape(layers))
    thicknesses = np.zeros(2 ** num_layers)  # get all possible thicknesses
    for i in range(2 ** num_layers):
        thickness = 0
        for j in range(num_layers):
            if i % (2 ** (num_layers - j)) // 2 ** (num_layers - j - 1) != 0:
                thickness += layers[j]
        thicknesses[i] = thickness

    return [phase(freqs, thickness, deltan) for thickness in thicknesses]  # get all possible phase shifts


def get_closest_phases(phases, ref_phase):
    for i in range(max(np.shape(phases))):
        phase = phases[i]
        for freq_nr in range(len(phase)):
            if phase[freq_nr] - ref_phase[freq_nr] > 0.5:
                phase[freq_nr] -= 1
            elif ref_phase[freq_nr] - phase[freq_nr] > 0.5:
                phase[freq_nr] += 1
        phases[i] = phase
    return phases

def objective(freqs, layers, deltan, number_of_tests=3e2):
    """
    Returns the RMS of the distance between a point in the phase space and the closest available space.
    Needs to be minimised
    """
    if number_of_tests is None:
        number_of_tests = int(3e2)
    else:
        number_of_tests = int(number_of_tests)

    nf = len(freqs)
    phases = get_all_phases(freqs, layers, deltan)

    obj = 0 # initialization
    for i in range(number_of_tests):
        test_phase = np.array([random.random() for i in range(nf)])
        closest_phases = get_closest_phases(phases, test_phase)

        obj += min([np.linalg.norm(test_phase - phas) for phas in closest_phases])**2 # add the smallest distance
    return obj / number_of_tests

def get_temperature(i, iterations):
    """
    Temperature in function od the steps
    """
    return 0.001*np.exp(-4 * i / iterations)

random.seed = 111
layers = np.array([random.randint(0, int(max_thickness*1e3))/1e3 for i in range(num_layers)])
current_obj = objective(frequencies, layers, deltan)


iterations = int(1e3)
# iterations = 1
plt.figure()
for i in range(iterations):
    T = get_temperature(i, iterations)
    change_layer = random.randint(0, num_layers-1)

    difference = [step_size*(1-random.randint(0, int(2*step_size*1e3)) / (1e3*step_size)) for i in range(num_layers)]
    # old_thickness = layers[change_layer]
    # layers[change_layer] = random.randint(0, int(max_thickness*1e3)) / 1e3 # change layer thickness on nm accuracy
    new_layers = [min(max(layers[i] + difference[i], 0), max_thickness) for i in range(num_layers)]

    new_obj = objective(frequencies, new_layers, deltan, number_of_tests=1e3 if T < 0.1 else None)
    # new_obj = 5
    # print(np.exp((new_obj - current_obj) / T))
    if new_obj <= current_obj or np.exp((current_obj - new_obj) / T) > random.random():
    # if current_obj - new_obj > 0:

        print(current_obj - new_obj)
        current_obj = new_obj
        layers = new_layers
        plt.clf()
        phases = np.array(get_all_phases(frequencies, layers, deltan))
        plt.scatter(phases[:, 0], phases[:, 1])
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        # plt.show()
        plt.title("Objective = " + str(current_obj))
        plt.pause(0.00001)


    if 10*i % iterations == 0:
        print(str(int(i/iterations*100)) + " %")

print(np.sqrt(current_obj))
print(layers)

with open("best_results_freq_" + str(frequencies) + "_layers_" + str(num_layers) + ".txt", 'a') as var_file:
    var_file.write("objective \t" + str(current_obj) + "\n")
    var_file.write("best_design \t" + str(layers) + "\n")

if len(frequencies) == 2:
    plt.figure()
    phases = np.array(get_all_phases(frequencies, layers, deltan))
    plt.scatter(phases[:, 0], phases[:, 1])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel("Phase shift for blue [/360°]")
    plt.ylabel("Phase shift for red [/360°]")
    plt.show()






