import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.sparse.csgraph import floyd_warshall

from dataFunctions import *

##Python implementation of the Isomap algorithm. Uses numpy, matplotlib, Floyd-Warshall implementation from SciPy,
#and function collection "dataFunctions.py".


#Since we are pretending that we are given a matrix of distances, data X will not be normalized
X,names,types = readDataFile("zoo.data")
X_norm = normalizeFeature(X,12) #Normalize the only numerical feature to be (approximately) in the same range as the booleans (0 to 1)
delta = getDistanceMatrix(X_norm)
''' delta = getDistanceMatrix(X) '''

#Set the p nearest neighbours
p=8
G=getGraph(delta,p)

#Retrieve the geodesic distance matrix from Graph using Floyd-Warshall
D=floyd_warshall(G,directed=False)
assert(np.array_equal(D.transpose(),D))

#Retrieve S using double centering
S_approx = doubleCentering(D)

#Calculate low-dimensional data matrix using classical MDS
Y,Eigs= MDS(S_approx,2)
Eigs=np.real(Eigs) #might contain very small complex parts due to numerical errors (S is approximated)
Y=Y.transpose()

#Plot results
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.scatter(Y[:,0],Y[:,1],c=types,)
for i,txt in enumerate(types):
    plt.annotate(str(int(txt)),[Y[i,0],Y[i,1]])
plt.show()