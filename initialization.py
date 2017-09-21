# Created on Sun Aug 27 13:11:12 2017
# @author: tom

import numpy as np
import clustering as cl
import dataset_io as dio
import grid
import fremen as fm


def whole_initialization(path, k, edge_of_square, timestep, longest, shortest,
                         radius):
    """
    input: path string, path to file
           k positive integer, number of clusters
           edge_of_square float, spatial edge of cell in default units (meters)
           timestep float, time edge of cell in default units (seconds)
           longest float, legth of the longest wanted period in default
                          units
           shortest float, legth of the shortest wanted period
                           in default units
           radius float, size of radius of the first found hypertime circle
    output: input_coordinates numpy array, coordinates for model creation
            overall_sum number (np.float64 or np.int64), sum of all measures
            structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
            C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
            shape_of_grid numpy array dx1 int64, number of cells in every
                                                 dimension
            time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
            T numpy array shape_of_grid[0]x1, time positions of timeframes
            W numpy array Lx1, sequence of reasonable frequencies
            ES float64, squared sum of squares of residues from this iteration
    uses: first_structure(), first_clustering(), grid.time_space_positions(),
          first_time_frame_probs(), fm.build_frequencies(), fm.chosen_period()
    objective: to perform first iteration step and to initialize variables
    """
    print('starting learning iteration: 0 (initialization)')
    structure = first_structure(path)
    C, U = first_clustering(path, k, structure)
    input_coordinates, time_frame_sums, overall_sum, shape_of_grid, T =\
        grid.time_space_positions(edge_of_square, timestep, path)
    time_frame_probs = first_time_frame_probs(overall_sum, shape_of_grid)
    W = fm.build_frequencies(longest, shortest)
    ES = -1  # no previous error
    P, W, ES = fm.chosen_period(T, time_frame_sums,
                                time_frame_probs, W, ES)
    structure[1].append(radius)
    structure[2].append(P)
    print('structure: ', structure)
    print('leaving learning iteration: 0 (initialization)')
    return input_coordinates, overall_sum, structure, C,\
        U, shape_of_grid, time_frame_sums, T, W, ES


def first_time_frame_probs(overall_sum, shape_of_grid):
    """
    input: overall_sum number (np.float64 or np.int64), sum of all measures
           shape_of_grid numpy array dx1 int64, number of cells in every
                                                dimension
    output: time_frame_probs numpy array shape_of_grid[0]x1, sum of
                                                             probabilities
                                                             over every
                                                             timeframe
    uses: np.array()
    objective: to create first time_frame_probs, i.e. time frames of a model
               that do not count the time
    """
    return np.array([overall_sum / shape_of_grid[0]] * shape_of_grid[0])


def first_clustering(path, k, structure):
    """
    input: path string, path to file
           k positive integer, number of clusters
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
    uses: dio.loading_data(), cl.k_means()
    objective: to create C and U for clusters initialization in next iteration
    """
    X = dio.loading_data(path)[:, 1:]
    # d = np.shape(X)[1]
    C, U, densities = cl.k_means(X, k, structure,
                                      method='random',  # initialization
                                      version='hard',  # objective function
                                      fuzzyfier=1,  # weighting exponent
                                      iterations=200,
                                      C_in=0, U_in=0)
    return C, U


def first_structure(path):
    """
    input: None
    output: structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
    uses: np.shape(), dio.loading_data()
    objective: to create initial structure
    """
    dim = np.shape(dio.loading_data(path))[1] - 1
    return [dim, [], []]
