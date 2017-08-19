# Created on Mon Jul 17 15:14:57 2017
# @author: tom


#    """
#    input:
#    output:
#    uses:
#    objective:
#    """
#
import pandas as pd
import numpy as np


def loading_file(data_location='' +
                 '/home/tom/projects/atomousek/stroll_fremen_nd/priklad.txt',
                 min_hours=-1, max_hours=25,
                 min_x=-8, max_x=13, min_y=-4, max_y=18,
                 expansion=True):
    """
    input: file_location string, predefined
           min_hours float, start of the measure
           max_hours float, finish of the measure
           min_x float, start of the measure
           max_x float, finish of the measure
           min_y float, start of the measure
           max_y float, finish of the measure
           expansion boolean, add two more points at the start and end of day
    output: X numpy array nx3, matrix of selected measures
    uses: pd.read_csv(), np.float64(), pd.loc[], pd.iloc[], pd.values()
    objective: load file, naturalize variables
    """
    # read DataFrame
    df = pd.read_csv(data_location, sep=' ', header=None, index_col=None)
    # naming columns
    df.columns = ['unix_time', 'x', 'y']
    # variables for naturalization
    df['hours'] = np.float64(df['unix_time'] % (60 * 60 * 24)) / (60 * 60)
    if expansion:
        # usualy I need whole day so I add random point at the end and start
        # of a day
        last = df.index.max()
        # random = np.random.random_sample()
        # x = (max_x - min_x) * random + min_x
        # y = (max_y - min_y) * random + min_y
        df.loc[last + 1] = [0.0, max_x, max_y, 0.0]
        df.loc[last + 2] = [0.0, min_x, min_y, 24.0]
        df.loc[last + 3] = [0.0, min_x, max_y, 0.0]
        df.loc[last + 4] = [0.0, max_x, min_y, 24.0]
    df = df.loc[(df['hours'] >= min_hours) & (df['hours'] <= max_hours) &
                (df['x'] >= min_x) & (df['x'] <= max_x) &
                (df['y'] >= min_y) & (df['y'] <= max_y), :]
    return df.iloc[:, 1:4].values


def number_of_edges(X, edge_of_square=0.05, timestep=0.1):
    """
    input: X numpy array nxd, matrix of measures
           edge_of_square float, length of the edge of 2D part of a "cell"
           timestep float, length of the time edge of a "cell"
    output: number_of_edges numpy array, number of edges on x, y and t axis
    uses:
    objective:
    """
    # number of predefined cubes in the measured space
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])
    t_min = np.min(X[:, 2])
    t_max = np.max(X[:, 2])
    number_of_cubes = [(x_max - x_min) / edge_of_square,
                       (y_max - y_min) / edge_of_square,
                       (t_max - t_min) / timestep]
    return np.int64(np.ceil(number_of_cubes))


def create_grid(X, number_of_cubes):
    """
    input:
    output:
    uses:
    objective: create 2D grid
    """
    H, edges = np.histogramdd(X, bins=number_of_cubes,
                              range=None, normed=False, weights=None)
    central_points = []
    for i in range(len(edges)):
        step_lenght = (edges[i][-1] - edges[i][0]) / len(edges[i])
        central_points.append(edges[i][0: -1] + step_lenght / 2)
    return H, central_points


def cartesian_product(*arrays):
    """
    downloaded from:
    'https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of'+\
    '-x-and-y-array-points-into-single-array-of-2d-points'
    input:
    output:
    uses:
    objective:
    """
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la],
                   dtype=arrays[0].dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def coordinates(central_points):
    """
    input:
    output:
    uses:
    objective:
    """
    return cartesian_product(*central_points)


def save_coordinates(edge_of_square=0.05, timestep=0.1,
                     save_directory='/home/tom/projects/atomousek/' +
                     'stroll_fremen_nd/output/',
                     data_file='/home/tom/projects/atomousek/' +
                     'stroll_fremen_nd/priklad.txt',
                     min_hours=-1, max_hours=25,
                     min_x=-8, max_x=12, min_y=-4, max_y=17,
                     expansion=False,
                     suffix=''
                     ):
    """
    input:
    output:
    uses:
    objective:
    """
    X = loading_file(data_file, min_hours, max_hours, min_x, max_x, min_y,
                     max_y, expansion)
    shape_of_grid = number_of_edges(X, edge_of_square, timestep)
    training_data, central_points = create_grid(X, shape_of_grid)
    positions = coordinates(central_points)
    training_data = training_data.reshape(-1)
    pd.DataFrame(positions).to_csv(path_or_buf=save_directory +
                                   'COORDINATES' + suffix,
                                   sep=' ', index=False, header=False)
    pd.DataFrame(shape_of_grid).to_csv(path_or_buf=save_directory +
                                       'SHAPE' + suffix,
                                       sep=' ', index=False, header=False)
    pd.DataFrame(training_data).to_csv(path_or_buf=save_directory +
                                       'DATA' + suffix,
                                       sep=' ', index=False, header=False)


def load_coordinates(coordinates_directory='/home/tom/projects/atomousek/' +
                     'stroll_fremen_nd/output/', suffix=''):
    """
    input:
    output:
    uses:
    objective:
    """
    coordinates = pd.read_csv(coordinates_directory + 'COORDINATES' + suffix,
                              sep=' ', header=None, index_col=None)
    shape_of_grid = pd.read_csv(coordinates_directory + 'SHAPE' + suffix,
                                sep=' ', header=None, index_col=None)
    training_data = pd.read_csv(coordinates_directory + 'DATA' + suffix,
                                sep=' ', header=None, index_col=None)
    return coordinates.values, shape_of_grid.values.reshape(-1),\
        training_data.values



















