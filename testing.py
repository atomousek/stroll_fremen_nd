# Created on Wed Aug 30 16:46:21 2017
# @author: tom


"""
tady budu spoustet time_space_positions(), histogram_probs() a np.histogramdd()
nad ruzne velkymi bunkami (postupne se budou zmensovat) a hledat situaci, kdy
ten model jeste dava dobre vysledky (soucet pravdepodobnosti v oblasti odpovida
poctu namerenych bodu). Tim najdu nejjemnejsi deleni prostoru. Budu tvrdit, ze
cim jemnejsi, tim lepsi model (???)
"""
import numpy as np
from time import clock

import clustering as cl
import dataset_io as dio
import fremen as fm
import grid
import initialization as init
import learning as lrn
import model as mdl

###########################
# only during developement
#import importlib
#importlib.reload(cl)
#importlib.reload(dio)
#importlib.reload(fm)
#importlib.reload(grid)
#importlib.reload(init)
#importlib.reload(lrn)
#importlib.reload(mdl)
##########################

# optional :)
#C = dio.load_numpy_array('k50_dva_cele_dny_C')
#COV = dio.load_numpy_array('k50_dva_cele_dny_COV')
#densities = dio.load_numpy_array('k50_dva_cele_dny_densities')
#structure = dio.load_list('k50_dva_cele_dny_structure')
#k = dio.load_list('k50_dva_cele_dny_k')


def iteration_over_space(path, C, COV, densities, structure, k):
    """
    input:
    output:
    uses:
    objective:
    """
    max_diff = 144
    default_shape = np.int64(np.array([1, 1, 1]))
    all_differences = []
    for i in range(1, max_diff + 1):
        start = clock()
        shape_of_grid = default_shape * i
        difference = model_measurement_differences(shape_of_grid, path, C, COV,
                                                   densities, structure, k)
        all_differences.append(difference)
        finish = clock()
        print('iteration: ', i)
        print('processor time: ', finish - start)
        print('actual difference: ', difference)
    output = np.array(all_differences)
    dio.save_numpy_array(output, 'k50_dva_cele_dny_differences')
    return output


def model_measurement_differences(shape_of_grid, path, C, COV, densities,
                                  structure, k):
    """
    input:
    output:
    uses:
    objective:
    """
    input_coordinates, histogram, overall_sum, shape_of_grid =\
        time_space_positions(shape_of_grid, path)
    hist_probs = histogram_probs(input_coordinates, C, COV, densities,
                                 structure, k, shape_of_grid, overall_sum)
    differences = histogram - hist_probs
    print('sum of measurements: ', np.sum(np.abs(histogram)))
    print('sum of probabilities: ', np.sum(np.abs(hist_probs)))
    random = np.random.rand(*shape_of_grid)
    random2 = random * overall_sum / np.sum(random)
    print('random difference: ', np.sum(np.abs(histogram - random2)))
    # maybe it would be good to see differences - we will see
    return np.sum(np.abs(differences))


def time_space_positions(shape_of_grid, path):
    """
    input: shape_of_grid numpy array int64
           path string, path to file
    output: input_coordinates numpy array, coordinates for model creation
            histogram numpy array shape_of_grid, sum of measures
                                                            over every
                                                            cell
            overall_sum number (np.float64 or np.int64), sum of all measures
            shape_of_grid
            T numpy array shape_of_grid[0]x1, time positions of measured values
    uses: loading_data(), number_of_edges(), hist_params(),
          cartesian_product()
    objective: to find central positions of cels of grid
    """
    data = dio.loading_data(path)
    central_points, histogram, overall_sum =\
        hist_params(data, shape_of_grid)
    return grid.cartesian_product(*central_points), histogram,\
        overall_sum, shape_of_grid


def hist_params(X, shape_of_grid):
    """
    input: X numpy array nxd, matrix of measures
           shape_of_grid numpy array dx1 int64, number of cells in every
                                                dimension
    output: central_points list (floats), central points of cells
            histogram numpy array shape_of_grid, sum of measures
                                                            over every
                                                            cell
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
    return central_points, histogram, np.sum(histogram)


def histogram_probs(input_coordinates, C, COV, densities, structure, k,
                    shape_of_grid, overall_sum):
    """
    input: input_coordinates numpy array, coordinates for model creation
           C numpy array kxd, matrix of k d-dimensional cluster centres
           COV numpy array kxdxd, matrix of covariance matrices
           densities numpy array kx1, matrix of number of measurements
                                      belonging to every cluster
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           k positive integer, number of clusters
           shape_of_grid numpy array dx1 int64, number of cells in every
                                                dimension
           overall_sum number (np.float64 or np.int64), sum of all measures
    output: hist_probs numpy array, 3D histogram of probabilities over grid
    uses: iter_over_probs(), np.reshape()
    objective: to create grid of probabilities over time-space
    """
    probs = mdl.iter_over_probs(input_coordinates, C, COV, densities,
                                structure, k)
    true_probs = probs * overall_sum / np.sum(probs)
    return true_probs.reshape(shape_of_grid)



































































