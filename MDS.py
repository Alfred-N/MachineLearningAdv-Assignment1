import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataFunctions import *

## Python implementation of the classical multidimensional scaling algorithm. Uses numpy, matplotlib and function collection "dataFunctions.py".


#Since we are pretending that we are given a matrix of distances, data X will not be normalized
X,names,types = readDataFile("zoo.data")
X_norm = normalizeFeature(X,12) 
X_centered = centerData(X)
X_c_and_norm = normalizeFeature(X_centered,12)

X_data=X_norm
D = getDistanceMatrix(X_data)
S_approx = doubleCentering(D)

Y,Eigs= MDS(S_approx,2)
Eigs=np.real(Eigs) #might contain very small complex parts due to numerical errors (S is approximated)
Y=Y.transpose()

#Plot results
plt.scatter(Y[:,0],Y[:,1],c=types,)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
for i,txt in enumerate(types):
    plt.annotate(str(int(txt)),[Y[i,0],Y[i,1]])
plt.title("$\sigma_1$ = " + str(round(Eigs[0]/np.mean(Eigs),3)) + "%" + "$\sigma_2$ = " + str(round(Eigs[1]/np.mean(Eigs),3)) + "%")
plt.show()
