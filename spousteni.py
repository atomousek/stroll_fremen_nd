# Created on Wed Aug 30 16:54:05 2017
# @author: tom


import numpy as np
from time import clock

import clustering as cl
import dataset_io as dio
import fremen as fm
import grid
import initialization as init
import learning as lrn
import model as mdl
import testing as tst
###########################
# only during developement
import importlib
importlib.reload(cl)
importlib.reload(dio)
importlib.reload(fm)
importlib.reload(grid)
importlib.reload(init)
importlib.reload(lrn)
importlib.reload(mdl)
importlib.reload(tst)
##########################

path = '/home/tom/projects/atomousek/stroll_fremen_nd/tri_tydny.txt'
longest = 60*60*24*28
shortest = 60*60*12
edge_of_square = 0.2  # 0.05
timestep = 720  # 300
k = 9
hours_of_measurement = 24 * 21

C, COV, densities, structure, k =\
    lrn.method(longest, shortest, path, edge_of_square, timestep, k,
               hours_of_measurement)



# patek, testovaci data
path_test = '/home/tom/projects/atomousek/stroll_fremen_nd/kontrolni_patek.txt'
hours_of_measurement = 24
statistics = tst.iteration_over_space(path_test, C, COV, densities, structure, k,
                                  edge_of_square, timestep,
                                  hours_of_measurement)

#
#dio.save_list(statistics, 'k30timestep300edge0.05covariances')
#
#
#
#path_test = '/home/tom/projects/atomousek/stroll_fremen_nd/tri_tydny.txt'
#
#statistics = tst.iteration_over_space(path_test, C, COV, densities, structure, k,
#                                  edge_of_square, timestep)
#


## ctvrtek, cast ucicich dat (pro srovnani)
#path_train = '/home/tom/projects/atomousek/stroll_fremen_nd/prvni_den.txt'
#
#statistics_train = tst.iteration_over_space(path_train, C, COV, densities, structure, k,
#                                  edge_of_square, timestep)



#
#statistics = []
#for edge_of_square in [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
#    print('spoustim edge_of_square = ', edge_of_square)
#    C, COV, densities, structure, k, stats =\
#        lrn.method(longest, shortest, path, edge_of_square, timestep, k)
#    statistics.append(stats)
#
#dio.save_list(statistics, 'edge_of_square_1._.5_..._.001_dva_cele_dny_stats')

#dio.save_numpy_array(C, 'k3_dva_cele_dny_C')
#dio.save_numpy_array(COV, 'k3_dva_cele_dny_COV')
#dio.save_numpy_array(densities, 'k3_dva_cele_dny_densities')
#
#dio.save_list(structure, 'k3_dva_cele_dny_structure')
#dio.save_list(k, 'k3_dva_cele_dny_k')
#dio.save_list(stats, 'k3_dva_cele_dny_stats')

# statistiky = tst.iteration_over_space(path, C, COV, densities, structure, k)

















