# Created on Sun Aug 27 14:40:40 2017
# @author: tom


import numpy as np
import model as mdl
import fremen as fm
import dataset_io as dio

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.misc import toimage


def time_frame_psti(longest, shortest,  # added by user
                    input_coordinates, overall_sum, structure, path, C_old,
                    U_old, k, shape_of_grid, time_frame_sums, amplitudes):
    """
#    input: X numpy array nxd, hyper space to analyze
#           input_coordinates numpy array, coordinates for model creation
#           overall_sum number (np.float64 or np.int64), sum of all measures
#           structure list(?), description of dimensions (which dim is time)
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
    jdi_dal = 0
    hist_probs, C, U = mdl.model_fremen(input_coordinates, overall_sum,
                                        structure, path, C_old, U_old, k,
                                        shape_of_grid)
    time_frame_probs = np.sum(hist_probs, axis=(1, 2))
    T, S = fm.residues(time_frame_sums, time_frame_probs)
    P, amplitude = fm.chosen_period(T, S, longest, shortest)
    amplitudes.append(amplitude)
    structure[1].append(P / (2 * np.pi))  # toto je spatne
    # jaky je vztah mezi P a novou dimenzi? kde to vlastne resim? fuck!
    # mozna budu muset premodelovat "structure" a krom polomeru tam dat i delky
    if len(amplitudes) < 3:  # hodne trapna podminka :)
        jdi_dal = 1
    if jdi_dal == 1:
        # zavolej sam sebe, ne? Nebo vytvorim funkci, ktera toto bude volat,
        # dokud tato funkce nevrati ukoncovaci priznak (return ..., 1)
        return C, U, amplitudes, structure
    else:
        # zavolej visualisation nebo neco takoveho
        hist_data = np.histogramdd(dio.loading_data(path), bins=shape_of_grid,
                                   range=None, normed=False, weights=None)[0]
        model_visualisation(hist_probs, hist_data, shape_of_grid)

# jak je to teda s tim shape_of_grid? prvni je casova dimenze???
# to se musi zohlednit ve vizualizaci ve for cyklu


def model_visualisation(H_probs, H_train, shape_of_grid):
    """
    input:
    output:
    uses:
    objective:
    """
#    H_probs = true_probabilities.reshape(shape_of_grid)
#    H_train = training_data.reshape(shape_of_grid)
    random_values = np.random.rand(*shape_of_grid)
    H_test = (random_values < H_probs) * 1
    # build pictures and save them
    fig = plt.figure(dpi=400)
    for i in range(shape_of_grid[2]):
        # training data
        plt.subplot(221)
        cmap = mpl.colors.ListedColormap(['white', 'blue', 'red'])
        bounds = [-0.5, 0.5, 1.5, 3000]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(H_train[:, :, i], interpolation='nearest',
                         cmap=cmap, norm=norm)
        plt.xticks([])
        plt.yticks([])
        # testing data
        plt.subplot(222)
        cmap = mpl.colors.ListedColormap(['white', 'blue', 'red'])
        bounds=[-0.5, 0.5, 1.5, 3000]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(H_test[:, :, i],interpolation='nearest',
                            cmap = cmap,norm=norm)
        plt.xticks([])
        plt.yticks([])
        # model
        plt.subplot(212)
        cmap = mpl.colors.ListedColormap(['black', 'blue', 'purple', 'red',
                                          'orange', 'pink', 'yellow', 'white'])
        bounds=[-0.5, 0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(H_probs[:, :, i],interpolation='nearest',
                            cmap = cmap,norm=norm)
        plt.colorbar(img, cmap=cmap,
                     norm=norm, boundaries=bounds, 
                     ticks=[0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 1],
                     fraction=0.046, pad=0.01)
        plt.xticks([])
        plt.yticks([])
        # all together
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        # name the file
        name = str(i)
        if len(name.split('.')[0]) == 1:
            name = '0' + name
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



























































































