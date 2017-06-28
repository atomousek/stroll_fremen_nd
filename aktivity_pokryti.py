# Created on Thu Jun 22 23:29:04 2017
# @author: tom


import importlib
import dataset_io as dio
import k_means as km
import numpy as np
import matplotlib.pyplot as plt


importlib.reload(dio)
importlib.reload(km)

X = dio.nacteni_prikladu()

k = 3


C, U = km.k_means(X, k, method='random', norm='M', version='fuzzy',
                  fuzzyfier=2, visualise=False, iterations=1000)



#plt.scatter(X[:, 2], X[:, 3], color='red', label='DATASET')
#plt.scatter(C[:, 2], C[:, 3], color='blue', label='CENTRA')
#plt.show()
#
#plt.scatter(X[:,0], X[:,1], color='red', label='DATASET')
#plt.scatter(C[:,0], C[:,1], color='blue', label='CENTRA')
#plt.show


C_300 = np.empty_like(C)
np.copyto(C_300, C)

# a ted se pokusim spocitat prumernou nejkratsi vzdalenost k centru
# coz bude norma pro vzdalenost k centru
# nebo spis se podivam na to rozdeleni tech minimalnich vzdalenosti,
# a pak se az rozhodnu, jakou miru zvolim
#
## vsechny vzdalenosti, nebo spis druhe mocnny vzdalenosti v kazde dimenzy
#D_300 = km.distance_matrix(X, C_300, U, norm='L2')
#
#U_300 = km.partition_matrix(D_300, version='hard', fuzzyfier=1)
#
#n, d = np.shape(X)
#
#all_diffs = np.array([[0, 0, 0, 0]])
#for centroid in range(k):
#    U_part = np.tile(U_300[centroid, :], (d, 1)).T
#    C_one = np.tile(C_300[centroid, :], (n, 1))
#    diffs = U_part * X - U_part * C_one
#    diffs = diffs[U_300[centroid, :] != 0, :]
#    all_diffs = np.r_[all_diffs, diffs]
#
#
## hist, bins = np.histogram(all_diffs[:, 3], bins=50)
## width = 0.7 * (bins[1] - bins[0])
## center = (bins[:-1] + bins[1:]) / 2
## plt.bar(center, hist, align='center', width=width)
## plt.show()
#
#np.std(all_diffs, axis=0)
#np.mean(all_diffs, axis=0)

# takze na to kaslu a ulozim ty koveriancni matice spolu s centry
X = dio.nacteni_prikladu()
k, n = np.shape(U)
X = np.tile(X, (k, 1, 1))
C = np.tile(C_300, (n, 1, 1))
C = np.transpose(C, [1, 0, 2])
XC = X - C  # X_knd - C_knd
COV = []
for cluster in range(k):
    # I am not sure, if it is ok to use X, as it was used before,
    # but I hope it will free the RAM (original X is probably large)
    XCp = XC[cluster, :, :]
    V = np.cov(XCp, aweights=U[cluster, :], rowvar=False)
    d = np.shape(V)[0]
    VD = V / (np.linalg.det(V) ** (1 / d))
    VI = np.linalg.inv(VD)
    COV.append(VI)

COV_300 = np.array(COV)


# takze ted mam 300 center a 300 koveriancnich matic
# kdyz vezmu jakoukoli hodnotu vektoru x (4d), ziskam jeji vzdalenost od 
# kterehokoli centra: d = (x-c_i).T * COV_i * (x-c_i)
# Mejme tedy matici X vsech vektoru x, potom vzdalenost od center spocitam:
#
# X = dio.nacteni_prikladu()  # sem musim dat ten grid
for cas in range(0, 241):
    cas_hod = cas / 10
    hodiny = np.ones((3000, 1)) * cas_hod# np.random.random_sample((30000, 1)) * 24
    x_y = np.random.random_sample((3000, 2)) * 2 - 1
    cos_t = np.cos(2*np.pi * hodiny / 24)
    sin_t = np.sin(2*np.pi * hodiny / 24)
    X_rand = np.c_[cos_t, sin_t, x_y]
    n = np.shape(X_rand)[0]
    X = np.tile(X_rand, (k, 1, 1))
    C = np.tile(C_300, (n, 1, 1))
    C = np.transpose(C, [1, 0, 2])
    XC = X - C  # X_knd - C_knd
    D = []
    for cluster in range(k):
        # I am not sure, if it is ok to use X, as it was used before,
        # but I hope it will free the RAM (original X is probably large)
        XCp = XC[cluster, :, :]
        VI = COV_300[cluster, :, :]
        D.append(np.sum(np.dot(XCp, VI) * XCp, axis=1))
    
    vzdalenosti = np.array(D)
    
    min_vzdalenosti = 1./np.min(vzdalenosti, axis=0)
    
    
    #np.savetxt('/home/tom/python/vyuka/numpy/aktivity/vzdalenosti.txt',
    #              vzdalenosti, fmt='%.18e', delimiter=' ', newline='\n')
    #
    #np.cos(2*np.pi * 12 / 24)
    #np.sin(2*np.pi * 12 / 24)
    #
    
    min_vzdalenosti[0]=50
    
    fig = plt.figure()
    # Plot...
    plt.scatter(X_rand[:,2], X_rand[:,3], c=min_vzdalenosti, s=200)
    plt.gray()
    
    #plt.show()
    fig.savefig('/home/tom/python/vyuka/numpy/aktivity/cas_' + str(cas_hod) + '.png')
    


















