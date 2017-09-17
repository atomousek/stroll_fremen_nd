# Created on Wed Sep 13 19:47:10 2017
# @author: tom


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

## optional :)
#C = dio.load_numpy_array('k50_dva_cele_dny_C')
#COV = dio.load_numpy_array('k50_dva_cele_dny_COV')
#densities = dio.load_numpy_array('k50_dva_cele_dny_densities')
#structure = dio.load_list('k50_dva_cele_dny_structure')
#k = dio.load_list('k50_dva_cele_dny_k')


def iteration_over_space(path, C, COV, densities, structure, k,
                         edge_of_square, timestep, hours_of_measurement,
                         prefix):
    """
    input:
    output:
    uses:
    objective:
    """
    hist_probs, input_coordinates, shape_of_grid =\
        histogram_of_probabilities(path, C, COV, densities, structure, k,
                                   edge_of_square, timestep)
    data = dio.loading_data(path)
    prumer = np.ones_like(hist_probs) * len(data) / len(input_coordinates)
    nula = np.zeros_like(hist_probs)
    t_max = int(shape_of_grid[0])
    x_max = int(shape_of_grid[1])
    y_max = int(shape_of_grid[2])
    prvni_kolo = 1
    diff_model = []
    diff_nuly = []
    diff_prumery = []
    while t_max >= 2:
        diff_m = []
        diff_n = []
        diff_p = []
        x_max = int(shape_of_grid[1])
        y_max = int(shape_of_grid[2])
        while min(x_max, y_max) >= 2:
            model = np.histogramdd(input_coordinates, bins=[t_max, x_max, y_max],
                                   range=None, normed=False, weights=hist_probs)[0]
            # !!! tady neni vyreseno, kdyz by ta data obsahovala hodnoty
            realita = np.histogramdd(data, bins=[t_max, x_max, y_max],
                                     range=None, normed=False, weights=None)[0]
            nuly = np.histogramdd(input_coordinates, bins=[t_max, x_max, y_max],
                                     range=None, normed=False, weights=nula)[0]
            prumery = np.histogramdd(input_coordinates, bins=[t_max, x_max, y_max],
                                     range=None, normed=False, weights=prumer)[0]
            diff = np.sum(np.abs(realita - model))
            if prvni_kolo == 1:
                lrn.model_visualisation(model, realita, [t_max, x_max, y_max],
                                        hours_of_measurement,
                                        prefix)
                prvni_kolo = 0
            print('shape of grid: ', t_max, ' ', x_max, ' ', y_max)
            print('realita minus model: ', diff)
            print('realita minus nuly: ', np.sum(np.abs(realita - nuly)))
            print('realita minus prumery: ', np.sum(np.abs(realita - prumery)))
            diff_m.append(diff)
            diff_n.append(np.sum(np.abs(realita - nuly)))
            diff_p.append(np.sum(np.abs(realita - prumery)))
            x_max = int(x_max / 2)
            y_max = int(y_max / 2)
        t_max = int(t_max / 2)
        diff_model.append(diff_m)
        diff_nuly.append(diff_n)
        diff_prumery.append(diff_p)
    ### shape of grid by pak byla nejaka promenna...
    return diff_model, diff_nuly, diff_prumery





def histogram_of_probabilities(path, C, COV, densities,
                               structure, k, edge_of_square, timestep):
    """
    input:
    output:
    uses:
    objective:
    """
    input_coordinates, overall_sum, shape_of_grid =\
        time_space_positions(edge_of_square, timestep, path)
    hist_probs = histogram_probs(input_coordinates, C, COV, densities,
                                 structure, k, overall_sum)
#    differences = histogram - hist_probs
#    print('sum of measurements: ', np.sum(np.abs(histogram)))
#    print('sum of probabilities: ', np.sum(np.abs(hist_probs)))
#    random = np.random.rand(*shape_of_grid)
#    random2 = random * overall_sum / np.sum(random)
#    print('random difference: ', np.sum(np.abs(histogram - random2)))
#    # maybe it would be good to see differences - we will see
#    return np.sum(np.abs(differences))
    return hist_probs, input_coordinates, shape_of_grid

##############################



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
            shape_of_grid
            T numpy array shape_of_grid[0]x1, time positions of measured values
    uses: loading_data(), number_of_edges(), hist_params(),
          cartesian_product()
    objective: to find central positions of cels of grid
    """
    data = dio.loading_data(path)
    shape_of_grid = number_of_cells(data, edge_of_square, timestep)
    print(shape_of_grid)
    central_points, overall_sum = hist_params(data, shape_of_grid)
    input_coordinates = cartesian_product(*central_points)
    return input_coordinates, overall_sum, shape_of_grid


def hist_params(data, shape_of_grid):
    """
    input: data numpy array nxd, matrix of measures
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
    histogram, edges = np.histogramdd(data, bins=shape_of_grid,
                                      range=None, normed=False, weights=None)
    central_points = []
    for i in range(len(edges)):
        step_lenght = (edges[i][-1] - edges[i][0]) / len(edges[i])
        central_points.append(edges[i][0: -1] + step_lenght / 2)
    overall_sum = np.sum(histogram)
    return central_points, overall_sum


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
    shape_of_grid = np.int64(np.ceil(number_of_cubes))
    return shape_of_grid


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




def histogram_probs(input_coordinates, C, COV, densities, structure, k,
                    overall_sum):
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
           overall_sum number (np.float64 or np.int64), sum of all measures
    output: hist_probs numpy array, 3D histogram of probabilities over grid
    uses: iter_over_probs(), np.reshape()
    objective: to create grid of probabilities over time-space
    """
    # puvodni densities nejsou pouzivany
    probs = mdl.iter_over_probs(input_coordinates, C, COV, densities,
                                structure, k, dense_calc=densities)
    return probs






def test_model(hist_probs, hist_data):
    """
    input:
    output:
    uses:
    objective:
    """
    t, x, y = np.shape(hist_data)
    max_dividing = min(int(min(t, x, y) / 2), 50)
    differences = []
    for dividing in range(1, max_dividing):
        start = clock()
        lengths = np.int64(np.ceil(np.array([t, x, y]) / dividing))
        difference = []
        for part_t in range(dividing):
            for part_x in range(dividing):
                for part_y in range(dividing):
                    difference.append(
                        np.sum(
                            hist_data[part_t * lengths[0]: (part_t + 1) * lengths[0],
                                      part_x * lengths[1]: (part_x + 1) * lengths[1],
                                      part_y * lengths[2]: (part_y + 1) * lengths[2]]
                            ) - np.sum(
                            hist_probs[part_t * lengths[0]: (part_t + 1) * lengths[0],
                                      part_x * lengths[1]: (part_x + 1) * lengths[1],
                                      part_y * lengths[2]: (part_y + 1) * lengths[2]]
                                    )
                                        )
        rozdil = np.sum(np.abs(np.array(difference)))
        finish = clock()
        print('deleni: ', dividing, ' rozdil: ', rozdil, ' cas: ', finish - start)
        differences.append(rozdil)
    return differences

###############################################

































