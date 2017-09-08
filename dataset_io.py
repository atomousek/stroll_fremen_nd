# Created on Fri Jun  2 13:52:33 2017
# @author: tom

import pandas as pd
import numpy as np
import pickle

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
    wavelengths = structure[2]
    X = np.empty((len(data), dim + len(radii) * 2))
    X[:, : dim] = data[:, 1: dim + 1]
    for period in range(len(radii)):
        r = radii[period]
        Lambda = wavelengths[period]
        X[:, dim: dim + 2] = np.c_[r * np.cos(data[:, 0] * 2 * np.pi / Lambda),
                                   r * np.sin(data[:, 0] * 2 * np.pi / Lambda)]
        dim = dim + 2
    # jeste si tu udelam vystup, ktery mi rekne, jaka je variabilita dat
    print('struktura prostoru: ', structure)
    print('kovarincni matice dat ve vytvorenem prostoru:')
    print(np.cov(X, ddof=0, rowvar=False))
    print('kovarincni matice dat v rozumnem zobrazeni:')
    XC = zobrazeni_do_rozumnych_souradnic(X, structure)
    print(np.cov(XC, ddof=0, rowvar=False))
    return X


def save_numpy_array(variable, name, save_directory='/home/tom/projects/' +
                     'atomousek/stroll_fremen_nd/output/variables/'):
    """
    input: variable numpy array, some variable
           name string, name of the file
           save_directory string, path to file, default 'variables'
    output: None
    uses: np.save()
    objective: to save numpy array to csv
    """
    if '.' in name:
        parts = name.rsplit('.')
        if parts[1] == '.npy':
            name = parts[0]
    np.save(save_directory + name + '.npy', variable)


def load_numpy_array(name, load_directory='/home/tom/projects/' +
                     'atomousek/stroll_fremen_nd/output/variables/'):
    """
    input: name string, name of the file
           load_directory string, path to file, default 'variables'
    output: variable numpy array, loaded variable
    uses: np.load()
    objective: to save numpy array
    """
    if '.' in name:
        parts = name.rsplit('.')
        if parts[1] == '.npy':
            name = parts[0]
    return np.load(load_directory + name + '.npy')


def save_list(variable, name, save_directory='/home/tom/projects/' +
              'atomousek/stroll_fremen_nd/output/variables/'):
    """
    input: variable numpy array, some variable
           name string, name of the file
           save_directory string, path to file, default 'variables'
    output: None
    uses: pickle.dump()
    objective: to save list
    """
    if '.' in name:
        parts = name.rsplit('.')
        if parts[1] == '.lst':
            name = parts[0]
    with open(save_directory + name + '.lst', mode='wb') as myfile:
        pickle.dump(variable, myfile)


def load_list(name, load_directory='/home/tom/projects/' +
              'atomousek/stroll_fremen_nd/output/variables/'):
    """
    input: name string, name of the file
           load_directory string, path to file, default 'variables'
    output: variable list (or int), loaded variable
    uses: pickle.load()
    objective: to load list
    """
    if '.' in name:
        parts = name.rsplit('.')
        if parts[1] == '.lst':
            name = parts[0]
    with open(load_directory + name + '.lst', mode='rb') as myfile:
        return pickle.load(myfile)


def zobrazeni_do_rozumnych_souradnic(X, structure):
    Ci = create_zeros(structure)
    observations = np.shape(X)[0]
    ones = np.ones((observations, 1))
    dim = structure[0]
    radii = structure[1]
    XC = np.empty((observations, dim + len(radii)))
    XC[:, : dim] = X[:, : dim] - Ci[:, : dim]
    # hypertime dimensions substraction
    for period in range(len(radii)):
        r = radii[period]
        cos = (np.sum(X[:, dim + (period * 2): dim + (period * 2) + 2] *
                      Ci[:, dim + (period * 2): dim + (period * 2) + 2],
                      axis=1, keepdims=True) / (r ** 2))
        cos = np.minimum(np.maximum(cos, ones * -1), ones)
        XC[:, dim + period: dim + period + 1] = r * np.arccos(cos)
    return XC


def create_zeros(structure):
    """
    input: path string, path to file
           structure list(int, list(floats)), number of non-hypertime
                                              dimensions and list of hypertime
                                              radii
    output: X numpy array nxd, matrix of measures in hypertime
    uses: loading_data(), np.empty(), np.c_[]
    objective: to create X as a data in hypertime
    """
    # pouzivam puvodni funkci pro vytvoreni "pocatku souradnic" v pocatku dne
    dim = structure[0]
    radii = structure[1]
    wavelengths = structure[2]
    data = np.zeros((1, dim + len(radii)))
    X = np.empty((1, dim + len(radii) * 2))
    X[:, : dim] = data[:, 1: dim + 1]
    for period in range(len(radii)):
        r = radii[period]
        Lambda = wavelengths[period]
        X[:, dim: dim + 2] = np.c_[r * np.cos(data[:, 0] * 2 * np.pi / Lambda),
                                   r * np.sin(data[:, 0] * 2 * np.pi / Lambda)]
        dim = dim + 2
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
