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

path = '/home/tom/projects/atomousek/stroll_fremen_nd/dva_cele_dny.txt'
longest = 60*60*24*14
shortest = 60*60
edge_of_square = 0.05
timestep = 300
k = 50


C, COV, densities, structure, k =\
    lrn.method(longest, shortest, path, edge_of_square, timestep, k)


dio.save_numpy_array(C, 'k50_dva_cele_dny_C')
dio.save_numpy_array(COV, 'k50_dva_cele_dny_COV')
dio.save_numpy_array(densities, 'k50_dva_cele_dny_densities')

dio.save_list(structure, 'k50_dva_cele_dny_structure')
dio.save_list(k, 'k50_dva_cele_dny_k')




















