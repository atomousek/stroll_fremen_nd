# Created on Thu Jun 22 23:29:04 2017
# @author: tom

#import dataset_io as dio
#import k_means as km
#import density as dy
import grid
import model

#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#from scipy.misc import toimage
###########################
## only during developement
#import importlib
#importlib.reload(dio)
#importlib.reload(km)
#importlib.reload(dy)
#importlib.reload(grid)
#importlib.reload(model)
##########################

grid.save_coordinates(edge_of_square=1, timestep=1)

coordinates, shape_of_grid, data = grid.load_coordinates()

true_probabilities = model.model_0(input_coordinates=coordinates, training_data=data, k=10)

model.model_visualisation(true_probabilities, data, shape_of_grid)



