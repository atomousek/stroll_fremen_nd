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
import testing_0d as tst0
import testing_0d_as_classes as tst0c
import testing_1d as tst1
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
importlib.reload(tst0)
importlib.reload(tst0c)
importlib.reload(tst1)
##########################


#############################
# dva tydny vs. dva a pul dne
#############################
path = '/home/tom/projects/atomousek/stroll_fremen_nd/trenovaci_dva_tydny.txt'

# takhle byly vygenerovany matice pro porovnani (hypercasu 2, r dne 2)
longest = 60*60*24*28
shortest = 60*60*12
edge_of_square = 0.15625*4#0.15625#1#0.2
timestep = 60*60*0.1875*16#60*20#60*60*0.24
k = 9
hours_of_measurement = 24 * 14 # nepotrebne

## generovani peknych obrazku
#longest = 60*60*24*28
#shortest = 60*60*12
#edge_of_square = 0.5#1#0.2
#timestep = 60*20#60*20#60*60*0.24
#k = 9
#hours_of_measurement = 24 * 14 # nepotrebne

C, COV, densities, structure, k =\
    lrn.method(longest, shortest, path, edge_of_square, timestep, k,
               hours_of_measurement)


importlib.reload(tst)  # predelat testovani granularity
# dva apul dne, testovaci data
path_test = '/home/tom/projects/atomousek/stroll_fremen_nd/testovaci_dva_dny.txt'
hours_of_measurement = 24*2
diff_model_test, diff_nuly_test, diff_prumery_test = tst.iteration_over_space(path_test, C, COV, densities, structure, k,
                                  edge_of_square, timestep,
                                  hours_of_measurement, prefix='dva_dny_testing_data')

#with open('/home/tom/projects/atomousek/stroll_fremen_nd/output/variables/statistics' +
#                      '.txt', 'w') as file:
#    file.write(str(diff_model_test))

# !!!!!!!!!!!!!!
# musim vytvorit datasety pro predikci, zatim se to resi jen na trenovacich datech

#############################################
# otevirani a zavirani dveri, pouze pozitivni
#############################################
## positives only binarni
#path = '/home/tom/projects/atomousek/stroll_fremen_nd/binarni_data_dvere_positives.txt'
#
#longest = 60*60*24*28
#shortest = 60*60*4
#edge_of_square = 0.2
#timestep = 1
#k = 9  # muzeme zkusit i 3
#hours_of_measurement = 24 * 14  # nepotrebne
#
#C, COV, densities, structure, k =\
#    lrn.method(longest, shortest, path, edge_of_square, timestep, k,
#               hours_of_measurement)
#
## positives only binarni
#importlib.reload(tst0)  # neni potreba testovat granularitu
## vsechny casy dveri, testovaci data
#path_test_times = '/home/tom/projects/atomousek/stroll_fremen_nd/binarni_data_dvere_all_times.txt'
#path_test_values = '/home/tom/projects/atomousek/stroll_fremen_nd/binarni_data_dvere_all_values.txt'
#hours_of_measurement = 24*7
#diff_model_test, diff_nuly_test, diff_prumery_test =\
#    tst0.iteration_over_space(path_test_times, path_test_values, C, COV, densities, structure, k,
#                                  edge_of_square, timestep,
#                                  hours_of_measurement, prefix='otevirani_dveri_positives_testing_data')
#

###################################################
# otevirani a zavirani dveri, pozitivni i negativni
###################################################

##positves and negatives binarni
#path_p = '/home/tom/projects/atomousek/stroll_fremen_nd/binarni_data_dvere_positives.txt'
#path_n = '/home/tom/projects/atomousek/stroll_fremen_nd/binarni_data_dvere_negatives.txt'
#
#longest = 60*60*24*28
#shortest = 60*60*4
#edge_of_square = 1  # nepotrebna hodnota
#timestep = 1
#k = 9  # muzeme zkusit i 9
#hours_of_measurement = 24 * 7  # nepotrebne
#
##positves and negatives binarni
#C_p, COV_p, densities_p, structure_p, k_p =\
#    lrn.method(longest, shortest, path_p, edge_of_square, timestep, k,
#               hours_of_measurement)
#C_n, COV_n, densities_n, structure_n, k_n =\
#    lrn.method(longest, shortest, path_n, edge_of_square, timestep, k,
#               hours_of_measurement)
#
##positves and negatives binar
#importlib.reload(tst0c)  # neni potreba testovat granularitu
## vsechny casy dveri, testovaci data
#path_test_times = '/home/tom/projects/atomousek/stroll_fremen_nd/binarni_data_dvere_all_times.txt'
#path_test_values = '/home/tom/projects/atomousek/stroll_fremen_nd/binarni_data_dvere_all_values.txt'
#hours_of_measurement = 24*7
#diff_model_test, diff_nuly_test, diff_prumery_test =\
#    tst0c.iteration_over_space(path_test_times, path_test_values,
#                               C_p, COV_p, densities_p, structure_p, k_p,
#                               C_n, COV_n, densities_n, structure_n, k_n,
#                                  edge_of_square, timestep,
#                                  hours_of_measurement, prefix='otevirani_dveri_p_n_testing_data')
#
#
#


#############################
# casy prujezdu
#############################
#path = '/home/tom/projects/atomousek/stroll_fremen_nd/casy_prujezdu_train.txt'

# takhle to resilo denni (tydenni?) periody (nepouzito
## 3 periody ?
#longest = 60*60*24*7
#shortest = 60*60*12
#edge_of_square = 1
#timestep = 60*60*6
#k = 9
#hours_of_measurement = 24 * 14 # nepotrebne


## reseni celeho datasetu (vcetne testovacich dat)
#path = '/home/tom/projects/atomousek/stroll_fremen_nd/casy_prujezdu_train.txt'
#longest = 60*60*24*7
#shortest = 60*60*2
#edge_of_square = 1
#timestep = 60*60
#k = 9
#hours_of_measurement = 24 * 14 # nepotrebne
#
#
#C, COV, densities, structure, k =\
#    lrn.method(longest, shortest, path, edge_of_square, timestep, k,
#               hours_of_measurement)
#
#
#importlib.reload(tst1)  # predelat/zrusit? testovani granularity
## dva apul dne, testovaci data
#path_test = '/home/tom/projects/atomousek/stroll_fremen_nd/casy_prujezdu_test.txt'
#hours_of_measurement = 24*3 # espravna hodnota (jen pro zobrazovani do videa
#diff_model_test, diff_nuly_test, diff_prumery_test = tst1.iteration_over_space(path_test, C, COV, densities, structure, k,
#                                  edge_of_square, timestep,
#                                  hours_of_measurement, prefix='casy_prujezdu_testing_data')

############# cela data (nepouzito)

#path_test = '/home/tom/projects/atomousek/stroll_fremen_nd/casy_prujezdu.txt'
#hours_of_measurement = 24*3 # espravna hodnota (jen pro zobrazovani do videa
#diff_model_test, diff_nuly_test, diff_prumery_test = tst1.iteration_over_space(path_test, C, COV, densities, structure, k,
#                                  edge_of_square, timestep,
#                                  hours_of_measurement, prefix='casy_prujezdu_all_data')



# !!!!!!!!!!!!!!!!

































########################
# nejay stary odpad
########################
# path = '/home/tom/projects/atomousek/stroll_fremen_nd/tri_tydny.txt'
#
#importlib.reload(tst)
## patek, testovaci data
#path_test = '/home/tom/projects/atomousek/stroll_fremen_nd/kontrolni_patek.txt'
#hours_of_measurement = 24
#diff_model_test, diff_nuly_test, diff_prumery_test = tst.iteration_over_space(path_test, C, COV, densities, structure, k,
#                                  edge_of_square, timestep,
#                                  hours_of_measurement, prefix='patek_testing_data')
#
## ctvrtek, cast ucicich dat (pro srovnani)
#path_train = '/home/tom/projects/atomousek/stroll_fremen_nd/kontrolni_ctvrtek.txt'
#
#diff_model_train, diff_nuly_train, diff_prumery_train = tst.iteration_over_space(path_train, C, COV, densities, structure, k,
#                                  edge_of_square, timestep,
#                                  hours_of_measurement, prefix='ctvrtek_training_data')
#
##################################
#
#
## vymyslena sobota
#path_train = '/home/tom/projects/atomousek/stroll_fremen_nd/vymyslena_sobota.txt'
#
#statistics_train = tst.iteration_over_space(path_train, C, COV, densities, structure, k,
#                                  edge_of_square, timestep,
#                                  hours_of_measurement = 48, prefix='sobota_testing_data')
#
#
#statistics = []
#for edge_of_square in [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
#    print('spoustim edge_of_square = ', edge_of_square)
#    C, COV, densities, structure, k, stats =\
#        lrn.method(longest, shortest, path, edge_of_square, timestep, k)
#    statistics.append(stats)
#
#dio.save_list(statistics, 'edge_of_square_1._.5_..._.001_dva_cele_dny_stats')
#
#dio.save_numpy_array(C, 'k9_tri_tydny_C')
#dio.save_numpy_array(COV, 'k9_tri_tydny_COV')
#dio.save_numpy_array(densities, 'k9_tri_tydny_densities')
#
#dio.save_list(structure, 'k9_tri_tydny_structure')
#dio.save_list(k, 'k9_tri_tydny_k')
#dio.save_list(stats, 'k3_dva_cele_dny_stats')

# statistiky = tst.iteration_over_space(path, C, COV, densities, structure, k)

















