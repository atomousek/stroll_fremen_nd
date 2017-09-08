# Created on Sun Aug 27 14:40:40 2017
# @author: tom


import numpy as np
from time import clock

import model as mdl
import fremen as fm
import dataset_io as dio
import initialization as init
# import grid
# import testing as tst

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.misc import toimage


def method(longest, shortest, path, edge_of_square, timestep, k,
           hours_of_measurement):
    """
    input:
    output:
    uses:
    objective:
    """
    # first of all, we need to find structure - we do not need high resolution
    # initialization
    k_puvodni = k  # nekompatibilni s py2.7, myslim
    input_coordinates, overall_sum, structure, C,\
        U, k, shape_of_grid, time_frame_sums, amplitudes, T, W =\
        init.whole_initialization(path, k, edge_of_square,
                                  timestep, longest, shortest)
    # iteration
    jump_out = 0
    iteration = 0
    while jump_out == 0:
        start = clock()
#        C, U, amplitudes, structure, hist_probs, hist_data, jump_out, COV,\
#            densities, W =\
        structure, jump_out, C, U, COV, densities, hist_probs, hist_data =\
            iteration_step(longest, shortest, path,  # added by user
                           input_coordinates, overall_sum, structure, C,
                           U, k, shape_of_grid, time_frame_sums, amplitudes, T, W)
        finish = clock()
        print('structure: ', structure)
#        print('densities: ', list(densities))
        iteration += 1
        print('learning iteration: ', iteration)
        print('processor time: ', finish - start)
    # some output
    print('learning iterations finished')#, visualisation started')
    # stats = tst.test_model(hist_probs, hist_data)
#    model_visualisation(hist_probs, hist_data, shape_of_grid,
#                        hours_of_measurement, prefix = 'training_data')
    # pravdepodobne bych mel ty parametry modelu nekam ukladat
    #
    # a tady teprve spocitam model (C_old a U_old jsou k nicemu - doufam ;) )
#    print('now it is time to build a model...')
#    C_old = 0
#    U_old = 0
#    C, U, COV, densities = mdl.model_parameters(path, structure, C_old, U_old,
#                                                k_puvodni)
    return C, COV, densities, structure, k_puvodni#, stats  # , hist_probs, hist_data


def iteration_step(longest, shortest, path,  # added by user
                   input_coordinates, overall_sum, structure, C_old,
                   U_old, k, shape_of_grid, time_frame_sums, amplitudes, T, W):
    """
#    input: X numpy array nxd, hyper space to analyze
#           input_coordinates numpy array, coordinates for model creation
#           overall_sum number (np.float64 or np.int64), sum of all measures
#           structure list(int, list(floats), list(floats)),
#                      number of non-hypertime dimensions, list of hypertime
#                      radii nad list of wavelengths
#           C_old numpy array kxd, centres from last iteration
#           k positive integer, number of clusters
#           shape_of_grid numpy array dx1 int64, number of cells in every
#                                                dimension
#    output: time_frame_probs numpy array shape_of_grid[0]x1, sum of
#                                                             probabilities
#                                                             over every
#                                                             timeframe
#            C numpy array kxd, matrix of k d-dimensional cluster centres
#    uses: model_fremen(), np.reshape(), np.sum()
    objective: to decide whether to visualize or iterate
    """
    jump_out = 0
    hist_probs, C, U, COV, densities =\
        mdl.model_fremen(input_coordinates, overall_sum,
                         structure, path, C_old, U_old, k, shape_of_grid)
    time_frame_probs = np.sum(hist_probs, axis=(1, 2))
    S = fm.residues(time_frame_sums, time_frame_probs)
    print('soucet chyb: ', np.sum(np.abs(S)))
    P, amplitude, W = fm.chosen_period(T, S, longest, shortest, W)
    # jaky je vztah mezi P a novou dimenzi? kde to vlastne resim? fuck!
    # mozna budu muset premodelovat "structure" a krom polomeru tam dat i delky
    if len(amplitudes) < 2:  # hodne trapna podminka :)
#        if P in structure[2]:
#            structure[1][structure[2].index(P)] = structure[1][structure[2].index(P)] * 2
#        else:
        amplitudes.append(amplitude)
#        structure[1].append(4)  # konstantni polomer pro vsechny dimenze
        structure[1].append(1)  # pokus odvozeny od rovnomerneho rozdeleni
        structure[2].append(P)
        # trochu zbesile, ale nepotrebuji to k nicemu az na konci a je to
        # pravdepodobne dost velke
        hist_probs = 0
        hist_data = 0
        COV = 0
        # densities = 0
    else:
        jump_out = 1
        # zavolej visualisation nebo neco takoveho
        hist_data = np.histogramdd(dio.loading_data(path), bins=shape_of_grid,
                                   range=None, normed=False, weights=None)[0]
#    return C, U, amplitudes, structure, hist_probs, hist_data, jump_out,\
#        COV, densities, W
    return structure, jump_out, C, U, COV, densities, hist_probs, hist_data


def model_visualisation(H_probs, H_train, shape_of_grid, hours_of_measurement,
                        prefix):
    """
    input:
    output:
    uses:
    objective:
    """
    #hours_of_measurement = 24  # because of name
#    H_probs = true_probabilities.reshape(shape_of_grid)
#    H_train = training_data.reshape(shape_of_grid)
    random_values = np.random.rand(*shape_of_grid)
    H_test = (random_values < H_probs) * 1
    # build pictures and save them
    fig = plt.figure(dpi=100)#(dpi=400)
    for i in range(shape_of_grid[0]):
        # training data
        plt.subplot(221)
        cmap = mpl.colors.ListedColormap(['white', 'blue', 'red'])
        bounds = [-0.5, 0.5, 1.5, 3000]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(H_train[i, :, :], interpolation='nearest',
                         cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        # testing data
        plt.subplot(222)
        cmap = mpl.colors.ListedColormap(['white', 'blue', 'red'])
        bounds=[-0.5, 0.5, 1.5, 3000]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(H_test[i, :, :],interpolation='nearest',
                            cmap = cmap,norm=norm)
        plt.xticks([])
        plt.yticks([])
        # model
        plt.subplot(212)
        cmap = mpl.colors.ListedColormap(['black', 'blue', 'purple', 'red',
                                          'orange', 'pink', 'yellow', 'white'])
        bounds=[-0.5, 0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(H_probs[i, :, :],interpolation='nearest',
                            cmap = cmap,norm=norm)
        plt.colorbar(img, cmap=cmap,
                     norm=norm, boundaries=bounds, 
                     ticks=[0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 1],
                     fraction=0.046, pad=0.01)
        plt.xticks([])
        plt.yticks([])
        # all together
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        # name the file, assuming thousands of files
        # firstly hours
        times = i / (shape_of_grid[0] / hours_of_measurement)
        hours = times % 24
        days = int(times / 24)
        hours = str(hours)
        days = str(days)
        if len(hours.split('.')[0]) == 1:
            hours = '0' + hours
        if len(hours.split('.')[1]) == 1:
            hours = hours + '0'
        if len(hours.split('.')[1]) > 2:
            hours = hours.split('.')[0] + '.' + hours.split('.')[1][:2]
        if len(days.split('.')[0]) == 1:
            days = '0' + days
        name = str(i) + '.' + days + '.' + hours
        if len(name.split('.')[0]) == 1:
            name = '0' + name
        if len(name.split('.')[0]) == 2:
            name = '0' + name
        if len(name.split('.')[0]) == 3:
            name = '0' + name
        name = prefix + name
        path = '/home/tom/projects/atomousek/stroll_fremen_nd/output/' +\
               'images/' + name + '.png'
        fig.canvas.draw()
        # really do not understand :) coppied from somewhere
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.clf()
        # save
        toimage(data).save(path)
    plt.close(fig)



























































































