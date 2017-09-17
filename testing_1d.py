# Created on Thu Sep 14 13:05:05 2017
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
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.misc import toimage

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
#    y_max = int(shape_of_grid[2])
    prvni_kolo = 1
#    prvni_kolo = 0
    diff_model = []
    diff_nuly = []
    diff_prumery = []
    while t_max >= 168:#2:
        diff_m = []
        diff_n = []
        diff_p = []
        x_max = int(shape_of_grid[1])
#        y_max = int(shape_of_grid[2])
#        while min(x_max, y_max) >= 2:
        while x_max >= 20:#2:
            model = np.histogramdd(input_coordinates, bins=[t_max, x_max],  #, y_max],
                                   range=None, normed=False, weights=hist_probs)[0]
            print('model:')
            print(model)
            with open('/home/tom/projects/atomousek/stroll_fremen_nd/output/variables/times_hour_distributions.txt', 'w') as file1:
                for i in range(len(model)):
                    file1.write(str(list(model[i])))
                    file1.write('\n')
            # !!! tady neni vyreseno, kdyz by ta data obsahovala hodnoty
            realita = np.histogramdd(data, bins=[t_max, x_max],  #, y_max],
                                     range=None, normed=False, weights=None)[0]
            print('realita:')
            print(realita)
            with open('/home/tom/projects/atomousek/stroll_fremen_nd/output/variables/times_hour_measures.txt', 'w') as file2:
                for j in range(len(realita)):
                    file2.write(str(list(realita[j])))
                    file2.write('\n')
            nuly = np.histogramdd(input_coordinates, bins=[t_max, x_max],  #, y_max],
                                     range=None, normed=False, weights=nula)[0]
            prumery = np.histogramdd(input_coordinates, bins=[t_max, x_max],  #, y_max],
                                     range=None, normed=False, weights=prumer)[0]
            diff = np.sum(np.abs(realita - model))
#            diff = np.sum((realita - model) ** 2)
            if prvni_kolo == 1:
                model_visualisation(model, realita, [t_max, x_max],#, y_max],
                                        hours_of_measurement,
                                        prefix)
                prvni_kolo = 0
            print('shape of grid: ', t_max, ' ', x_max)  #, ' ', y_max)
            print('realita minus model: ', diff)
            print('realita minus nuly: ', np.sum(np.abs(realita - nuly)))
            print('realita minus prumery: ', np.sum(np.abs(realita - prumery)))
            diff_m.append(diff)
            diff_n.append(np.sum(np.abs(realita - nuly)))
            diff_p.append(np.sum(np.abs(realita - prumery)))
#            diff_n.append(np.sum((realita - nuly) ** 2))
#            diff_p.append(np.sum((realita - prumery) ** 2))
            x_max = int(x_max / 2)
#            y_max = int(y_max / 2)
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
    # hist_probs = probs * overall_sum / np.sum(probs)
    # return hist_probs
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



############################################
# zobrazovani

def zobrazeni_odhadu(model, realita, krok=60*60*24*14):
    """
    blbiny
    input: krok int, length of the step
           t numpy array, vector of targets
           y numpy array of the same length as y, vector of predictions
    output: none
    uses: matplotlib.pyplot.*np.c_
    objective: show series of grph devided by the krok
    """
    import matplotlib.pyplot as plt
    for i in range(0, len(model), krok):
        plt.plot(realita[i:i+krok], color='blue', label='realita')
        plt.plot(model[i:i+krok], color='red', label='model')
        plt.title('otevirani a zavirani dveri')
        plt.xlabel('cas')
        plt.ylabel('otevreno/zavreno')
        plt.legend()
        plt.show()
        plt.close()





def model_visualisation(model, realita, shape_of_grid, hours_of_measurement,
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
#    random_values = np.random.rand(*shape_of_grid)
#    H_test = (random_values < H_probs) * 1
    # build pictures and save them
    fig = plt.figure(dpi=400)
    # training data
    plt.subplot(121)
    cmap = mpl.colors.ListedColormap(['black', 'red', 'orange', 'yellow'])
    bounds = [-0.5, 0.64, 1.28, 2.56, 5.12]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    img = plt.imshow(realita[:, :], interpolation='nearest',
                     cmap=cmap, norm=norm)
    plt.xticks([])
    plt.yticks([])
#    # testing data
#    plt.subplot(222)
#    cmap = mpl.colors.ListedColormap(['black', 'red', 'orange', 'pink', 'yellow', 'white', 'lightblue'])
#    bounds=[-0.5, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 3000]
#    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#    img = plt.imshow(H_test[i, :, :],interpolation='nearest',
#                        cmap = cmap,norm=norm)
#    plt.xticks([])
#    plt.yticks([])
    plt.xticks([0, 6, 12, 18])
    plt.yticks([0, 24, 48, 72, 96, 120, 144, 168])
    # model
    plt.subplot(122)
#    cmap = mpl.colors.ListedColormap(['black', 'blue', 'purple', 'red',
#                                      'orange', 'pink', 'yellow', 'white', 'lightblue'])
    cmap = mpl.colors.ListedColormap(['black', 'white', 'lightblue', 'blue', 'purple', 'red',
                                      'orange', 'yellow'])
    bounds=[-0.5, 0.02, 0.04, 0.08, 0.32, 0.64, 1.28, 2.56, 5.12]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    img = plt.imshow(model[:, :],interpolation='nearest',
                        cmap = cmap,norm=norm)
    plt.colorbar(img, cmap=cmap,
                 norm=norm, boundaries=bounds, 
                 ticks=[0.02, 0.04, 0.08, 0.32, 0.64, 1.28, 2.56, 5.12],
                 fraction=0.046, pad=0.01)
    plt.xticks([0, 6, 12, 18])
    plt.yticks([0, 24, 48, 72, 96, 120, 144, 168])
    # all together
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    # name the file, assuming thousands of files
    # firstly hours
#    times = i / (shape_of_grid[0] / hours_of_measurement)
#    hours = times % 24
#    days = int(times / 24)
#    hours = str(hours)
#    days = str(days)
#    if len(hours.split('.')[0]) == 1:
#        hours = '0' + hours
#    if len(hours.split('.')[1]) == 1:
#        hours = hours + '0'
#    if len(hours.split('.')[1]) > 2:
#        hours = hours.split('.')[0] + '.' + hours.split('.')[1][:2]
#    if len(days.split('.')[0]) == 1:
#        days = '0' + days
#    name = str(i) + '.' + days + '.' + hours
#    if len(name.split('.')[0]) == 1:
#        name = '0' + name
#    if len(name.split('.')[0]) == 2:
#        name = '0' + name
#    if len(name.split('.')[0]) == 3:
#        name = '0' + name
#    name = prefix + name
    name = prefix
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































