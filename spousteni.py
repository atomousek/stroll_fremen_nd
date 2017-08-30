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

path = '/home/tom/projects/atomousek/stroll_fremen_nd/prvni_den.txt'
longest = 60*60*12
shortest = 60*15
edge_of_square = 0.05
timestep = 300
k = 3


C, COV, densities, structure, k =\
    lrn.method(longest, shortest, path, edge_of_square, timestep, k)















































































