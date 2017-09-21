# Created on Sun Aug 27 14:40:40 2017
# @author: tom


import numpy as np
from time import clock

import model as mdl
import fremen as fm
import dataset_io as dio
import initialization as init


def method(longest, shortest, path, edge_of_square, timestep, k, radius,
           number_of_periods):
    """
    input: longest float, legth of the longest wanted period in default
                          units
           shortest float, legth of the shortest wanted period
                           in default units
           path string, path to file
           edge_of_square float, spatial edge of cell in default units (meters)
           timestep float, time edge of cell in default units (seconds)
           k positive integer, number of clusters
           radius float, size of radius of the first found hypertime circle
           number_of_periods int, max number of added hypertime circles
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
            COV numpy array kxdxd, matrix of covariance matrices
            density_integrals numpy array kx1, matrix of ratios between
                                               measurements and grid cells
                                               belonging to the clusters
            structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
    uses: time.clock()
          init.whole_initialization(), iteration_step()
    objective: to learn model parameters
    """
    # initialization
    input_coordinates, overall_sum, structure, C,\
        U, shape_of_grid, time_frame_sums, T, W, ES =\
        init.whole_initialization(path, k, edge_of_square,
                                  timestep, longest, shortest, radius)
    # iteration
    jump_out = 0
    iteration = 0
    while jump_out == 0:
        iteration += 1
        print('starting learning iteration: ', iteration)
        start = clock()
        structure, jump_out, C, U, COV, density_integrals, hist_probs, hist_data, W,\
            ES =\
            iteration_step(path,
                           input_coordinates, overall_sum, structure, C,
                           U, k, shape_of_grid, time_frame_sums, T,
                           W, ES, number_of_periods)
        finish = clock()
        print('structure: ', structure)
        print('leaving learning iteration: ', iteration)
        print('processor time: ', finish - start)
    print('learning iterations finished')
    return C, COV, density_integrals, structure


def iteration_step(path,
                   input_coordinates, overall_sum, structure, C_old,
                   U_old, k, shape_of_grid, time_frame_sums, T, W,
                   ES, number_of_periods):
    """
    input: path string, path to file
           input_coordinates numpy array, coordinates for model creation
           overall_sum number (np.float64 or np.int64), sum of all measures
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           C_old numpy array kxd, centres from last iteration
           U_old numpy array kxn, matrix of weights from the last iteration
           k positive integer, number of clusters
           shape_of_grid numpy array dx1 int64, number of cells in every
                                                dimension
           time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
           T numpy array shape_of_grid[0]x1, time positions of timeframes
           W numpy array Lx1, sequence of reasonable frequencies
           ES float64, squared sum of squares of residues from this iteration
           number_of_periods int, max number of added hypertime circles
#    output: time_frame_probs numpy array shape_of_grid[0]x1, sum of
#                                                             probabilities
#                                                             over every
#                                                             timeframe
#            C numpy array kxd, matrix of k d-dimensional cluster centres
#    uses: model_fremen(), np.reshape(), np.sum()
    objective: to decide whether to visualize or iterate
    """
    jump_out = 0
    hist_probs, C, U, COV, density_integrals =\
        mdl.model_fremen(input_coordinates, overall_sum,
                         structure, path, C_old, U_old, k, shape_of_grid)
    osy = tuple(np.arange(len(np.shape(hist_probs)) - 1) + 1)
    time_frame_probs = np.sum(hist_probs, axis=osy)
    P, W, ES = fm.chosen_period(T, time_frame_sums,
                                           time_frame_probs, W, ES)
    if len(structure[1]) < number_of_periods:
        structure[2].append(P)
        structure[1].append(structure[1][-1] *
                            structure[2][-2] / structure[2][-1])
        # we do not need these:
        hist_probs = 0
        hist_data = 0
        COV = 0
    else:
        jump_out = 1
        hist_data = np.histogramdd(dio.loading_data(path), bins=shape_of_grid,
                                   range=None, normed=False, weights=None)[0]
    return structure, jump_out, C, U, COV, density_integrals, hist_probs, hist_data,\
        W, ES
