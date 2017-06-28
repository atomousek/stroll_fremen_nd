# Created on Fri Jun  2 13:52:33 2017
# @author: tom

import pandas as pd
import numpy as np


def nacteni_prikladu(umisteni_souboru="" +
                     "/home/tom/projects/atomousek/stroll_fremen_nd/" +
                     "priklad.txt"):
    # nactu DataFrame
    trenovaci = pd.read_csv(umisteni_souboru, sep=' ', header=None,
                            index_col=None)
    # pojmenuji sloupce
    trenovaci.columns = ['cas', 'x', 'y']
    hodiny = np.float64(trenovaci['cas'] % (60 * 60 * 24)) / (60 * 60)
    x_min = np.min(trenovaci['x'])
    x_max = np.max(trenovaci['x'])
    x_avg = (x_min + x_max) / 2
    y_min = np.min(trenovaci['y'])
    y_max = np.max(trenovaci['y'])
    y_avg = (y_min + y_max) / 2

    trenovaci['cosinus_t'] = np.cos(2*np.pi * hodiny / 24)
    trenovaci['sinus_t'] = np.sin(2*np.pi * hodiny / 24)
    trenovaci['X_norm'] = 2 * (trenovaci['x'] - x_avg) / (x_max - x_min)
    trenovaci['Y_norm'] = 2 * (trenovaci['y'] - y_avg) / (y_max - y_min)
    return trenovaci.iloc[:, 3:7].values
