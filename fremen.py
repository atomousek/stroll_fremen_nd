# Created on Sat Aug 19 14:22:31 2017
# @author: tom


import numpy as np


def build_frequencies(longest, shortest):
    """
    input: longest float, legth of the longest (default) period in default
            units
           shortest float, legth of the shortest (usualy an hour) period
           in default units
           span float, time length of the measured sequence in default units
    output: W numpy array Lx1, sequence of frequencies
    uses: np.arange()
    objective: to find frequencies w_0 to w_k
    """
    k = int(longest / shortest) + 1
    return np.arange(k) / longest


def complex_numbers_batch(T, S, W):
    """
    input: T numpy array Nx1, time positions of measured values
           S numpy array Nx1, sequence of measured values
           W numpy array Lx1, sequence of reasonable frequencies
    output: G numpy array Lx1, sequence of complex numbers corresponding
            to the frequencies from W
    uses:
    objective: to find sparse(?) frequency spectrum of the sequence S
    """
    Gs = S * (np.e ** (W[:, np.newaxis] * T * (-1j) * np.pi * 2))
    return np.mean(Gs, axis=1)


def max_influence(W, G):
    """
    input: W numpy array Lx1, sequence of reasonable frequencies
           G numpy array Lx1, sequence of complex numbers corresponding
           to the frequencies from W
    output: P float64, length of the most influential frequency in default
            units
    uses: np.absolute(), np.argmax(), np.float64()
    objective: to find length of the most influential periodicity in default
               units
    """
    influential_frequency = W[np.argmax(np.absolute(G[1:])) + 1]
    if influential_frequency == 0:
        return np.float64(0.0)
    else:
        return 1 / influential_frequency


def chosen_period(T, S, longest, shortest):
    """
    input: T numpy array Nx1, time positions of measured values
           S numpy array Nx1, sequence of measured values
           longest float, legth of the longest (default) period in default
           units
           shortest float, legth of the shortest (usualy an hour) period
           in default units
    output: P float64, length of the most influential frequency in default
            units
            amplitude float64, max size of G
    uses:
    objective:
    """
    W = build_frequencies(longest, shortest)
    G = complex_numbers_batch(T, S, W)
    print('amplitudy: ', np.absolute(G[1:]))
    P = max_influence(W, G)
    amplitude = np.max(np.absolute(G[1:]))
    return P, amplitude


def residues(time_frame_sums, time_frame_probs):
    """
    input: time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
           time_frame_probs numpy array shape_of_grid[0]x1, sum of
                                                            probabilities
                                                            over every
                                                            timeframe
    output: S numpy array shape_of_grid[0]x1, sequence of measured values
    uses:
    objective: create dataset for fremen
    """
    return time_frame_sums - time_frame_probs




## tests
##T, S = create_random_sequence(N=100000)
#T = np.arange(30*24)
## S = np.tile(np.random.randint(2, size=5), 30)
#S = np.tile(np.array([5, 4, 4, 3, 4, 3, 2, 1, 4, 3, 3, 2, 3, 2, 2, 1, 4, 3, 2, 1, 2, 1, 1, 0]), 30)
#
#W = build_frequencies(24, 2)
#
#G = complex_numbers_batch(T, S, W)
#
##G = complex_numbers_batch(T, S - prumer, W)
#
#P = max_influence(W, G)
#
#tab, seq, deltas, prumer,full_reconstruction = reconstruction_test(T, G, S, W)
##tab_prum, seq_prum, deltas_prum, prumer_prum,full_reconstruction_prum = reconstruction_test(T, G, S - prumer, W)
##
##
##s = S[:, np.newaxis].T
##
##pokus = np.r_[s, seq]
##
##

# np.sum(full_reconstruction, axis=0)


#def create_random_sequence(N=100000):
#    """
#    input: N integer, number of elements
#    output: T numpy array Nx1, corresponding time positions
#            S numpy array Nx1, sequence of zeros and ones
#    uses: np.random.randint()
#    objective: to create random sequence for testing
#    """
#    return np.arange(N), np.random.randint(2, size=N)
#
#
#
#def reconstruction_test(T, G, S, W):
#    """
#    input: T numpy array Nx1, time positions of measured values
#           G numpy array Lx1, sequence of complex numbers corresponding
#           to the frequencies from W
#           S numpy array Nx1, sequence of measured values
#           W numpy array Lx1, sequence of reasonable frequencies
#    output:
#    uses:
#    objective: to reconstruct S using every one period and compare it with
#               delka periody, frekvence periody, velikost vlivu a
#               pomer velikosti vlivu a frekvence = soucin amplitudy a delky (!)
#    """
#    phis = np.angle(G)[1:, np.newaxis]
#    alphas = np.absolute(G)
#    alpha0 = alphas[0]
#    alphas = alphas[1:, np.newaxis]
#    ws = W[1:, np.newaxis]
#    sequences = alpha0 + alphas * np.cos(ws * np.pi * 2 * T + phis)
#    full_reconstruction = alpha0 + np.sum(alphas * np.cos(ws * np.pi * 2 * T + phis), axis=0)
#    deltas = np.abs(sequences - S)
#    errors = np.sum(deltas, axis=1)
#    velikosti = np.absolute(G)[1:, np.newaxis]
#    frekvence = ws
#    delky = 1/ws
#    pomer = velikosti / frekvence
#    return np.c_[frekvence, delky, velikosti, pomer, errors, phis], np.r_[S[:, np.newaxis].T, sequences], deltas, alpha0, np.c_[S, full_reconstruction, np.abs(S - full_reconstruction)]


























































































































