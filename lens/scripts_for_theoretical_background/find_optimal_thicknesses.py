"""
Simulated annealing optimization of the thicknesses of a frequency splitter.
"""

import numpy as np
import random
from matplotlib import pyplot as plt
import os

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
    """
    Makes sure distance between phases takes into account periodicity
    """
    closest_phases = phases
    for i in range(np.shape(phases)[0]):
        phase = phases[i]
        for freq_nr in range(len(phase)):
            if phase[freq_nr] - ref_phase[freq_nr] > 0.5:
                phase[freq_nr] -= 1
            elif ref_phase[freq_nr] - phase[freq_nr] > 0.5:
                phase[freq_nr] += 1
        closest_phases[i] = phase
    return closest_phases

def objective(freqs, layers, deltan, number_of_tests=3e3):
    """
    Returns the RMS of the distance between a point in the phase space and the closest available space.
    Needs to be minimised by the thermal anneal.
    """
    number_of_tests = int(number_of_tests)

    nf = len(freqs) # number of frequencies
    phases = get_all_phases(freqs, layers, deltan) # get all available phases

    obj = 0 # initialization
    for i in range(number_of_tests):
        test_phase = np.array([random.random() for i in range(nf)]) # generate random phase
        closest_phases = get_closest_phases(phases, test_phase) # take into account periodicity

        obj += min([np.linalg.norm(test_phase - phas) for phas in closest_phases])**power # add the smallest distance
    return (obj / number_of_tests)**(1/power) # mean of squares of average distances

def get_temperature(i, iterations):
    """
    Temperature in function of the steps.
    temperature decreases exponentially during anneal --> it gets increasingly difficult to increase the energy
    of the system
    """
    return 5e-3*np.exp(-4 * i / iterations)

if not os.path.exists("./best_layers"):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs("./best_layers")

max_thicknesses = [0.3, 0.4, 0.5, 0.6] # maximum thickness of a single layer

for num_layers in range(1, 7):
    for max_thickness in max_thicknesses:
        # num_layers = 4 # number of layers to optimize
        power = 2
        frequencies = [0.47, 0.55, 0.65] # frequencies that you want to optimise independently
        # frequencies = [0.6]


        deltan = 2.7 - 1.45 # difference in refractive index

        step_size = 0.1 # size of perturbation in every step


        seed = 111 # makes sure we can reproduce results --> change to get different results
        np.random.seed(seed)

        # initialize starting layer thicknesses and objective
        layers = np.array([random.randint(0, int(max_thickness*1e3))/1e3 for i in range(num_layers)])


        # layers = np.array([0.5, 0.278, 0.087, 0.42])
        # layers = np.array([0.356, 0.255])
        current_obj = objective(frequencies, layers, deltan)
        best_obj = current_obj
        best_layers = layers
        iterations = int(1e4)
        accuracy = int(2e2)
        plt.figure()
        for i in range(iterations):
            T = get_temperature(i, iterations)
            # change_layer = random.randint(0, num_layers-1)
            # old_thickness = layers[change_layer]
            # layers[change_layer] = random.randint(0, int(max_thickness*1e3)) / 1e3 # change layer thickness on nm accuracy

            # Perturbation on thicknesses:
            difference = [step_size*(1-random.randint(0, int(2*step_size*1e3)) / (1e3*step_size)) for i in range(num_layers)]
            new_layers = [min(max(layers[i] + difference[i], 0), max_thickness) for i in range(num_layers)]

            # new objective
            new_obj = objective(frequencies, new_layers, deltan, accuracy)

            # check if energy decreased OR accept increase with certain probability
            if new_obj <= current_obj or np.exp((current_obj - new_obj) / T) > random.random():

                print(current_obj - new_obj)
                # adjust new value
                current_obj = new_obj
                layers = new_layers

                if new_obj < best_obj:
                    best_obj = new_obj
                    best_layers = new_layers

                # plot
                # if len(frequencies) == 2:
                #     plt.clf()
                #     phases = np.array(get_all_phases(frequencies, layers, deltan))
                #     plt.scatter(phases[:, 0], phases[:, 1])
                #     plt.xlim((0, 1))
                #     plt.ylim((0, 1))
                #     plt.title("Objective = " + str(current_obj))
                #     plt.pause(0.00001)

            # print progress
            if 10*i % iterations == 0:
                print(str(int(i/iterations*100)) + " %")
                if (10*i)//iterations == 9:
                    step_size = 0.01

                accuracy = ((10*i)//iterations+1) * 200
                best_obj = objective(frequencies, new_layers, deltan, accuracy)
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_layers = layers



        # print result
        print(best_obj)
        print(best_layers)
        phases = np.array(get_all_phases(frequencies, best_layers, deltan))
        print(phases)

        # save result
        with open("./best_layers/best_results_"+ str(power) + "thorder_freq_" + str(frequencies) + "_layers_" + str(num_layers) + ".txt", 'a') as var_file:
            var_file.write("objective \t" + str(best_obj) + "\n")
            var_file.write("best_design \t" + str(best_layers) + "\n")
            var_file.write("phases \t" + str(phases) + "\n")

        # plot result
        if len(frequencies) == 2:
            plt.figure()
            plt.scatter(phases[:, 0], phases[:, 1])
            plt.xlim((0, 1))
            plt.ylim((0, 1))
            plt.xlabel("Phase shift for blue [/360°]")
            plt.ylabel("Phase shift for red [/360°]")
            plt.savefig("./best_layers/best_results_"+ str(power) + "thorder_freq_" + str(frequencies) + "_layers_" + str(num_layers) + "max_t" + str(max_thickness) + ".png")


