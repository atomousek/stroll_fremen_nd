# Created on Sun Aug 27 13:11:12 2017
# @author: tom

import numpy as np
import clustering as cl
import dataset_io as dio
import grid
import fremen as fm
# for fremen


def whole_initialization(path, k, edge_of_square, timestep, longest, shortest):
    """
    input:
    output:
    uses:
    objective:
    """
    amplitudes = first_amplitudes()
    structure = first_structure(path)
    C, U = first_clustering(path, k, structure)
    input_coordinates, time_frame_sums, overall_sum, shape_of_grid, T =\
        grid.time_space_positions(edge_of_square, timestep, path)
    time_frame_probs = first_time_frame_probs(overall_sum, shape_of_grid)
    S = fm.residues(time_frame_sums, time_frame_probs)
    print('soucet chyb: ', np.sum(np.abs(S)))
    W = fm.build_frequencies(longest, shortest)
    print('vsechny periody: ', list(1/W[1:]))
    P, amplitude, W = fm.chosen_period(T, S, longest, shortest, W)
    amplitudes.append(amplitude)
#    structure[1].append(4)  # konstantni polomer pro vsechny dimenze
    structure[1].append(4)  # pokus s velikostmi kruznic
    structure[2].append(P)
    print('structure: ', structure)
    return input_coordinates, overall_sum, structure, C,\
        U, k, shape_of_grid, time_frame_sums, amplitudes, T, W


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
    C, U, COV, densities = cl.k_means(X, k, structure,  # Gustafson–Kessel
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


def first_amplitudes():
    """
    input: None
    output: amplitudes list(float64), in thos case it is empty
    uses: None
    objective: return empty list of amplitudes
    """
    return []

























































































