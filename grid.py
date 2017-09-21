# Created on Mon Jul 17 15:14:57 2017
# @author: tom

"""

"""

import numpy as np
import dataset_io as dio


def time_space_positions(edge_of_square, timestep, path):
    """
    input: edge_of_square float, spatial edge of cell in meters
           timestep float, time edge of cell in seconds
           path string, path to file
    output: input_coordinates numpy array, coordinates for model creation
            time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
            overall_sum number (np.float64 or np.int64), sum of all measures
            shape_of_grid numpy array dx1 int64, number of cells in every
                                                 dimension
            T numpy array shape_of_grid[0]x1, time positions of measured values
    uses: loading_data(), number_of_edges(), hist_params(),
          cartesian_product()
    objective: to find central positions of cels of grid
    """
    data = dio.loading_data(path)
    shape_of_grid = number_of_cells(data, edge_of_square, timestep)
    central_points, time_frame_sums, overall_sum =\
        hist_params(data, shape_of_grid)
    input_coordinates = cartesian_product(*central_points)
    T = central_points[0]
    return input_coordinates, time_frame_sums, overall_sum,\
        shape_of_grid, T


def hist_params(X, shape_of_grid):
    """
    input: X numpy array nxd, matrix of measures
           shape_of_grid numpy array dx1 int64, number of cells in every
                                                dimension
    output: central_points list (floats), central points of cells
            time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
            overall_sum number (np.float64 or np.int64), sum of all measures
    uses: np.histogramdd()
    objective: find central points of cells of grid
    """
    histogram, edges = np.histogramdd(X, bins=shape_of_grid,
                                      range=None, normed=False, weights=None)
    central_points = []
    for i in range(len(edges)):
        step_lenght = (edges[i][-1] - edges[i][0]) / len(edges[i])
        central_points.append(edges[i][0: -1] + step_lenght / 2)
    #############################################################
    # FUNKCNI PRO 2d DATA !!!!
    # time_frame_sums = np.sum(histogram, axis=(1, 2))
    ##############################################################
    # pravdepodobne funkcni pro 1+ dimensionalni data
    osy = tuple(np.arange(len(np.shape(histogram)) - 1) + 1)
    time_frame_sums = np.sum(histogram, axis=osy)
    return central_points, time_frame_sums, np.sum(time_frame_sums)


def number_of_cells(X, edge_of_square, timestep):
    """
    input: X numpy array nxd, matrix of measures
           edge_of_square float, length of the edge of 2D part of a "cell"
           timestep float, length of the time edge of a "cell"
    output: shape_of_grid numpy array, number of edges on t, x, y, ... axis
    uses:np.shape(), np.max(), np.min(),np.ceil(), np.int64()
    objective: find out number of cells in every dimension
    """
    # number of predefined cubes in the measured space
    n, d = np.shape(X)
    number_of_cubes = [(np.max(X[:, 0]) - np.min(X[:, 0])) / timestep]
    for i in range(1, d):
        number_of_cubes.append((np.max(X[:, i]) - np.min(X[:, i])) /
                               edge_of_square)
    return np.int64(np.ceil(number_of_cubes))


def cartesian_product(*arrays):
    """
    downloaded from:
    'https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of'+\
    '-x-and-y-array-points-into-single-array-of-2d-points'
    input: *arrays enumeration of central_points
    output: numpy array (central positions of cels of grid)
    uses: np.empty(),np.ix_(), np.reshape()
    objective: to perform cartesian product of values in columns
    """
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la],
                   dtype=arrays[0].dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)
