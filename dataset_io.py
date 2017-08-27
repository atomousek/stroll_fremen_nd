# Created on Fri Jun  2 13:52:33 2017
# @author: tom

import pandas as pd
import numpy as np


# for fremen


def loading_data(path):
    """
    input: path string, path to file
    output: data numpy array nxd*, matrix of measures IRL
    uses: pd.read_csv(), pd.columns(), np.arange(), pd.values(), pd.loc[]
    objective: load data from file (t, x, y, ...)
    """
    df = pd.read_csv(path, sep=' ', header=None, index_col=None)
    dimension = len(df.columns)
    observations = len(df)
    if dimension == 1:
        df[1] = np.arange(observations)
        df.loc[:, ::-1]
    return df.values


def create_X(data, structure):
    """
    input: path string, path to file
           structure list(int, list(floats)), number of non-hypertime
                                              dimensions and list of hypertime
                                              radii
    output: X numpy array nxd, matrix of measures in hypertime
    uses: loading_data(), np.empty(), np.c_[]
    objective: to create X as a data in hypertime
    """
    dim = structure[0]
    radii = structure[1]
    X = np.empty((len(data), dim + len(radii) * 2))
    X[:, : dim] = data[:, 1: dim + 1]
    for period in range(len(radii)):
        r = radii[period]
        dim = dim + period * 2
        X[:, dim: dim + 2] = np.c_[r * np.cos(data[:, 0] / r),
                                   r * np.sin(data[:, 0] / r)]
    return X




#
#def loading_priklad_01(file_location='' +
#                          '/home/tom/projects/atomousek/stroll_fremen_nd/' +
#                          'priklad.txt', edge=0.05):
#    """
#    input: file_location string, defined as constant
#           edge float, length of the edge of "cell"
#    output: X numpy array nxd, matrix of normalized measures
#            number_of_cubes list, number of cubes on x and y axis
#    uses: pd.read_csv(), np.float64(), np.min(), np.max(), np.cos(), np.sin(),
#          pd.iloc(), pd.values()
#    objective: load file, normalize variables, return number of cubes
#    """
#    # read DataFrame
#    df = pd.read_csv(file_location, sep=' ', header=None, index_col=None)
#    # naming columns
#    df.columns = ['unix_time', 'x', 'y']
#    # variables for normalization
#    hours = np.float64(df['unix_time'] % (60 * 60 * 24)) / (60 * 60)
#    x_min = np.min(df['x'])
#    x_max = np.max(df['x'])
#    x_avg = (x_min + x_max) / 2
#    y_min = np.min(df['y'])
#    y_max = np.max(df['y'])
#    y_avg = (y_min + y_max) / 2
#    # number of predefined cubes in the measured space
#    number_of_cubes = [(x_max - x_min) / edge, (y_max - y_min) / edge]
#    # normalized variables
#    df['cosinus_t'] = np.cos(2*np.pi * hours / 24)
#    df['sinus_t'] = np.sin(2*np.pi * hours / 24)
#    df['X_norm'] = 2 * (df['x'] - x_avg) / (x_max - x_min)
#    df['Y_norm'] = 2 * (df['y'] - y_avg) / (y_max - y_min)
#    return df.iloc[:, 3:7].values, np.int64(np.ceil(number_of_cubes))
#
#
#def loading_priklad_natural(file_location='' +
#                            '/home/tom/projects/atomousek/stroll_fremen_nd/' +
#                            'priklad.txt', min_hours=-1, max_hours=25):
#    """
#    input: file_location string, defined as constant
#           edge float, length of the edge of "cell"
#    output: X numpy array nxd, matrix of normalized measures
#            number_of_cubes list, number of cubes on x and y axis
#    uses: pd.read_csv(), np.float64(), np.min(), np.max(), np.cos(), np.sin(),
#          pd.iloc(), pd.values()
#    objective: load file, naturalize variables, return number of cubes
#    """
#    # read DataFrame
#    df = pd.read_csv(file_location, sep=' ', header=None, index_col=None)
#    # naming columns
#    df.columns = ['unix_time', 'x', 'y']
#    # variables for naturalization
#    df['hours'] = np.float64(df['unix_time'] % (60 * 60 * 24)) / (60 * 60)
#    df = df.loc[(df['hours'] >= min_hours) & (df['hours'] < max_hours), :]
#    # naturalized variables
#    df['cosinus_t'] = (24 / (2*np.pi)) * np.cos(2*np.pi * df['hours'] / 24)
#    df['sinus_t'] = (24 / (2*np.pi)) * np.sin(2*np.pi * df['hours'] / 24)
#    df['X_norm'] = df['x']
#    df['Y_norm'] = df['y']
#    return df.iloc[:, 4:8].values
