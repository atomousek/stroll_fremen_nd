# Created on Fri Jun  2 13:52:33 2017
# @author: tom

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def nacteni_prvniho_souboru(umisteni_souboru="" +
                            "/home/tom/python/vyuka/numpy" +
                            "/training_data.txt"):
    # nactu DataFrame
    trenovaci = pd.read_csv(umisteni_souboru, sep=' ', header=None,
                            index_col=None)
    # pojmenuji sloupce
    trenovaci.columns = ['cas', 'hodnota']
    # trenovaci.head()
    # cas ale predzpracovat musim.
    #trenovaci['sekundy'] = trenovaci['cas'] % 60
    #trenovaci['minuty'] = np.floor((trenovaci['cas'] % (60 * 60)) / 60)
#    trenovaci['hodiny'] = np.floor((trenovaci['cas'] % (60 * 60 * 24)) /
#                                   (60 * 60))
#    trenovaci['dvouhodiny'] = np.floor((trenovaci['cas'] % (60 * 60 * 24)) /
#                                   (60 * 60 * 2))
#    trenovaci['ctyrhodiny'] = np.floor((trenovaci['cas'] % (60 * 60 * 24)) /
#                                   (60 * 60 * 4))
#    trenovaci['osmihodiny'] = np.floor((trenovaci['cas'] % (60 * 60 * 24)) /
#                                   (60 * 60 * 8))
#    trenovaci['dny'] = np.floor((trenovaci['cas'] % (60 * 60 * 24 * 7)) /
#                                (60 * 60 * 24))
    trenovaci['Rhodiny'] = (trenovaci['cas'] % (60 * 60 * 24)) / (60 * 60)
    trenovaci['Rdvouhodiny'] = ((trenovaci['cas'] % (60 * 60 * 24)) /
                                   (60 * 60 * 2))
    trenovaci['Rctyrhodiny'] = ((trenovaci['cas'] % (60 * 60 * 24)) /
                                   (60 * 60 * 4))
    trenovaci['Rosmihodiny'] = ((trenovaci['cas'] % (60 * 60 * 24)) /
                                   (60 * 60 * 8))
    trenovaci['Rdny'] = ((trenovaci['cas'] % (60 * 60 * 24 * 7)) /
                                (60 * 60 * 24))
#    trenovaci['novy_cas'] = trenovaci['cas']
    # trenovaci.tail()
    # a udelam z toho dve matice (array), X a y
    return trenovaci.iloc[:, 2:].values, trenovaci.iloc[:, 1].values


def zobrazeni_odhadu(t, y, krok=60*60*24):
    """
    input: krok int, length of the step
           t numpy array, vector of targets
           y numpy array of the same length as y, vector of predictions
    output: none
    uses: matplotlib.pyplot.*np.c_
    objective: show series of grph devided by the krok
    """
    for i in range(0, len(t), krok):
        plt.plot(t[i:i+krok], color='red', label='PUVODNI')
        plt.plot(y[i:i+krok], color='blue', label='ODHAD')
        plt.title('otevirani a zavirani dveri')
        plt.xlabel('cas')
        plt.ylabel('otevreno/zavreno')
        plt.legend()
        plt.show()


def nacteni_aktivit(umisteni_souboru="" +
                            "/home/tom/python/vyuka/numpy/aktivity" +
                            "/activity.txt"):
    # nactu DataFrame
    trenovaci = pd.read_csv(umisteni_souboru, sep=',', header=None,
                            index_col=None)
    # pojmenuji sloupce
    trenovaci.columns = ['None', 'Bed_to_Toilet', 'Eating', 'Enter_Home',
                         'Housekeeping', 'Leave_Home', 'Meal_Preparation',
                         'Relax', 'Resperate', 'Sleeping', 'Wash_Dishes',
                         'Work']
    trenovaci['cas'] = range(len(trenovaci['None']))
    trenovaci['minuty'] = np.float64(trenovaci['cas'] % (60))
    trenovaci['hodiny'] = np.float64(trenovaci['cas'] % (60 * 24)) / (60)
    trenovaci['dny'] = np.float64(trenovaci['cas'] % (60 * 24 * 7)) / (60 * 24)
    # OUTPUT: T, X
    return trenovaci.iloc[:, :12].values, trenovaci.iloc[:, 13:].values



def nacteni_prikladu(umisteni_souboru="" +
                            "/home/tom/python/vyuka/numpy" +
                            "/priklad.txt"):
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


# umisteni_souboru="/home/tom/python/vyuka/numpy/aktivity/activity.txt"