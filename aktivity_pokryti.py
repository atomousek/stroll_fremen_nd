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


# data save
#import pandas as pd
#df1 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/2017_05_25.txt', sep=' ', header=None, index_col=None)
#df2 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/2017_05_26.txt', sep=' ', header=None, index_col=None)
#df = pd.concat([df1, df2], ignore_index=True)
#
#pd.DataFrame(df.iloc[:, [0, 1, 2]]).to_csv('/home/tom/projects/atomousek/stroll_fremen_nd/dva_dny.txt',
#                                       sep=' ', index=False, header=False)
#pd.DataFrame(df1.iloc[:, [0, 1, 2]]).to_csv('/home/tom/projects/atomousek/stroll_fremen_nd/prvni_den.txt',
#                                       sep=' ', index=False, header=False)
#pd.DataFrame(df2.iloc[:, [0, 1, 2]]).to_csv('/home/tom/projects/atomousek/stroll_fremen_nd/druhy_den.txt',
#                                       sep=' ', index=False, header=False)
#import pandas as pd
#df1 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_05_23.txt', sep=';', header=None, index_col=None)
#df2 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_05_24.txt', sep=';', header=None, index_col=None)
#df3 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_05_25.txt', sep=';', header=None, index_col=None)
#df4 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_05_30.txt', sep=';', header=None, index_col=None)
#df5 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_05_31.txt', sep=';', header=None, index_col=None)
#df6 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_06_01.txt', sep=';', header=None, index_col=None)
#df7 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_06_06.txt', sep=';', header=None, index_col=None)
#df8 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_06_07.txt', sep=';', header=None, index_col=None)
#df9 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_06_08.txt', sep=';', header=None, index_col=None)
#df0 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_05_26.txt', sep=';', header=None, index_col=None)
#df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9], ignore_index=True)
#pd.DataFrame(df.iloc[:, [0, 1, 2]]).to_csv('/home/tom/projects/atomousek/stroll_fremen_nd/tri_tydny.txt',
#                                       sep=' ', index=False, header=False)
#pd.DataFrame(df0.iloc[:, [0, 1, 2]]).to_csv('/home/tom/projects/atomousek/stroll_fremen_nd/kontrolni_patek.txt',
#                                       sep=' ', index=False, header=False)


#import pandas as pd
#df1 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_05_23.txt', sep=';', header=None, index_col=None)
#df2 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_05_24.txt', sep=';', header=None, index_col=None)
#df3 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_05_25.txt', sep=';', header=None, index_col=None)
#df4 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_05_30.txt', sep=';', header=None, index_col=None)
#df5 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_05_31.txt', sep=';', header=None, index_col=None)
#df6 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_06_01.txt', sep=';', header=None, index_col=None)
#df7 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_06_06.txt', sep=';', header=None, index_col=None)
#df8 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_06_07.txt', sep=';', header=None, index_col=None)
#df9 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/data/people_detections_2017_06_08.txt', sep=';', header=None, index_col=None)
#
#df_train = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
#pd.DataFrame(df_train.iloc[:, [0, 1, 2]]).to_csv('/home/tom/projects/atomousek/stroll_fremen_nd/trenovaci_dva_tydny.txt',
#                                       sep=' ', index=False, header=False)
#df_test = pd.concat([df8, df9], ignore_index=True)
#df_test = df_test.loc[(df_test[1] >=-8) & (df_test[1] <=12) & (df_test[2] >=-3) & (df_test[2] <=17), :]
#pd.DataFrame(df_test.iloc[:, [0, 1, 2]]).to_csv('/home/tom/projects/atomousek/stroll_fremen_nd/testovaci_dva_dny.txt',
#                                       sep=' ', index=False, header=False)


#import pandas as pd
#df1 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/binarni_data_dvere.txt', sep=' ', header=None, index_col=None)
#df2 = df1.loc[df1[1] == 1, 0:0]
#df3 = df1.loc[:, 0:0]
#df4 = df1.loc[:, 1:1]
#df5 = df1.loc[df1[1] == 0, 0:0]
#df2.to_csv('/home/tom/projects/atomousek/stroll_fremen_nd/binarni_data_dvere_positives.txt',
#                                       sep=' ', index=False, header=False)
#df3.to_csv('/home/tom/projects/atomousek/stroll_fremen_nd/binarni_data_dvere_all_times.txt',
#                                       sep=' ', index=False, header=False)
#df4.to_csv('/home/tom/projects/atomousek/stroll_fremen_nd/binarni_data_dvere_all_values.txt',
#                                       sep=' ', index=False, header=False)
#df5.to_csv('/home/tom/projects/atomousek/stroll_fremen_nd/binarni_data_dvere_negatives.txt',
#                                       sep=' ', index=False, header=False)


#
#import pandas as pd
#df1 = pd.read_csv('/home/tom/projects/atomousek/stroll_fremen_nd/LabDoor2_LabDoor1.data', sep=' ', header=None, index_col=None)
#df2 = df1.loc[df1[2] == 1, 0:1]
#df2.to_csv('/home/tom/projects/atomousek/stroll_fremen_nd/casy_prujezdu.txt',
#                                       sep=' ', index=False, header=False)
#


# two days
grid.save_coordinates(edge_of_square=0.1, timestep=0.1,
                      data_file='/home/tom/projects/atomousek/' +
                      'stroll_fremen_nd/dva_dny.txt',
                       expansion=True, suffix='_0')



# first day
grid.save_coordinates(edge_of_square=0.1, timestep=0.1,
                      data_file='/home/tom/projects/atomousek/' +
                      'stroll_fremen_nd/prvni_den.txt',
                      expansion=True, suffix='_1')



# second day
grid.save_coordinates(edge_of_square=0.1, timestep=0.1,
                      data_file='/home/tom/projects/atomousek/' +
                      'stroll_fremen_nd/druhy_den.txt',
                      expansion=True, suffix='_2')





print('save done')
# model
coordinates_0, shape_of_grid_0, data_0 = grid.load_coordinates(suffix='_0')
coordinates_1, shape_of_grid_1, data_1 = grid.load_coordinates(suffix='_1')
coordinates_2, shape_of_grid_2, data_2 = grid.load_coordinates(suffix='_2')

true_probabilities_0 = model.model_0(input_coordinates=coordinates_0, training_data=data_0, k=30,
                                   data_location='/home/tom/projects/atomousek/stroll_fremen_nd/' +
                                   'dva_dny.txt')


import numpy as np
true_probabilities = true_probabilities_0 / 2
#data = np.empty((2 * len(data_0), 1))
#true_probabilities[:len(true_probabilities_0)] = true_probabilities_0
#true_probabilities[len(true_probabilities_0):] = true_probabilities_0
#true_probabilities = true_probabilities / 2
#data[:len(data_0), :] = data_1
#data[len(data_0):, :] = data_2
#shape_of_grid = np.array([200, 210, 480])

# visualisation
model.model_visualisation(true_probabilities, data_1, shape_of_grid_1)

# se to prepise!!!
model.model_visualisation(true_probabilities, data_2, shape_of_grid_2)

