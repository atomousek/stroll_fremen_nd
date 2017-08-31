# Created on Wed Aug 30 16:46:21 2017
# @author: tom


"""
tady budu spoustet time_space_positions(), histogram_probs() a np.histogramdd()
nad ruzne velkymi bunkami (postupne se budou zmensovat) a hledat situaci, kdy
ten model jeste dava dobre vysledky (soucet pravdepodobnosti v oblasti odpovida
poctu namerenych bodu). Tim najdu nejjemnejsi deleni prostoru. Budu tvrdit, ze
cim jemnejsi, tim lepsi model (???)
"""
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
##########################

# optional :)
#dio.load_numpy_array('k50_dva_cele_dny_C')
#dio.load_numpy_array('k50_dva_cele_dny_COV')
#dio.load_numpy_array('k50_dva_cele_dny_densities')
#dio.load_numpy_array('k50_dva_cele_dny_structure')
#dio.load_numpy_array('k50_dva_cele_dny_k')