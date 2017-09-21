# Created on Sat Aug 19 14:22:31 2017
# @author: tom

"""
basic FreMEn to find most influential periodicity, call
chosen_period(T, time_frame_sums, time_frame_probs, longest, shortest, W, ES):
it returns the most influential period in the timeseries, where timeseries are
    the residues between reality and model
where
input: T numpy array Nx1, time positions of measured values
       time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                        over every
                                                        timeframe
       time_frame_probs numpy array shape_of_grid[0]x1, sum of
                                                        probabilities in model
                                                        over every
                                                        timeframe
       W numpy array Lx1, sequence of reasonable frequencies
       ES float64, squared sum of squares of residues from the last
                   iteration
and
output: P float64, length of the most influential frequency in default
        units
        amplitude float64, max size of G
        W numpy array Lx1, sequence of reasonable frequencies without
                           the chosen one
        ES_new float64, squared sum of squares of residues from
                        this iteration

for the creation of a list of reasonable frequencies call
build_frequencies(longest, shortest):
where
longest - float, legth of the longest wanted period in default units, 
        - usualy four weeks
shortest - float, legth of the shortest wanted period in default units,
         - usualy one hour.
It is necessary to understand what periodicities you are looking for (or what
    periodicities you think are the most influential)
"""

import numpy as np


def build_frequencies(longest, shortest):
    """
    input: longest float, legth of the longest wanted period in default
                          units
           shortest float, legth of the shortest wanted period
                           in default units
    output: W numpy array Lx1, sequence of frequencies
    uses: np.arange()
    objective: to find frequencies w_0 to w_k
    """
    k = int(longest / shortest) + 1
    W = np.arange(k) / longest
    return W


def complex_numbers_batch(T, S, W):
    """
    input: T numpy array Nx1, time positions of measured values
           S numpy array Nx1, sequence of measured values
           W numpy array Lx1, sequence of reasonable frequencies
    output: G numpy array Lx1, sequence of complex numbers corresponding
            to the frequencies from W
    uses: np.e, np.newaxis, np.pi, np.mean()
    objective: to find sparse(?) frequency spectrum of the sequence S
    """
    Gs = S * (np.e ** (W[:, np.newaxis] * T * (-1j) * np.pi * 2))
    G = np.mean(Gs, axis=1)
    return G


def max_influence(W, G):
    """
    input: W numpy array Lx1, sequence of reasonable frequencies
           G numpy array Lx1, sequence of complex numbers corresponding
                              to the frequencies from W
    output: P float64, length of the most influential frequency in default
                       units
            W numpy array Lx1, sequence of reasonable frequencies without
                               the chosen one
    uses: np.absolute(), np.argmax(), np.float64(),np.array()
    objective: to find length of the most influential periodicity in default
               units and return changed list of frequencies
    """
    maximum_position = np.argmax(np.absolute(G[1:])) + 1
    print('pozice_vybraneho_W: ', maximum_position)
    print('hodnota vybraneho 1/W: ', 1/W[maximum_position])
    # ! probably not elegant way of changing W
    WW = list(W)
    influential_frequency = WW.pop(maximum_position)
    W = np.array(WW)
    # !
    if influential_frequency == 0:
        P = np.float64(0.0)
    else:
        P = 1 / influential_frequency
    return P, W


def chosen_period(T, time_frame_sums, time_frame_probs, W, ES):
    """
    input: T numpy array Nx1, time positions of measured values
           time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
           time_frame_probs numpy array shape_of_grid[0]x1, sum of
                                                            probabilities
                                                            over every
                                                            timeframe
           W numpy array Lx1, sequence of reasonable frequencies
           ES float64, squared sum of squares of residues from the last
                       iteration
    output: P float64, length of the most influential frequency in default
            units
            amplitude float64, max size of G
            W numpy array Lx1, sequence of reasonable frequencies without
                               the chosen one
            ES_new float64, squared sum of squares of residues from
                            this iteration
    uses: np.sum(), np.max(), np.absolute()
          complex_numbers_batch(), max_influence()
    objective: to choose the most influencing period in the timeseries, where
               timeseries are the residues between reality and model
    """
    S = time_frame_sums - time_frame_probs
    ES_new = (np.sum(S ** 2)) ** 0.5
    print('squared sum of squares of residues: ', ES_new)
    if ES == -1:
        print('difference in errors: ', ES_new)
    else:
        dES = ES_new - ES
        print('difference in errors: ', dES)
        if dES > 0:
            print('too many periodicities, choose less')
    G = complex_numbers_batch(T, S, W)
    P, W = max_influence(W, G)
    amplitude = np.max(np.absolute(G[1:]))
    return P, amplitude, W, ES_new
