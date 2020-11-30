import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataFunctions import readDataFile
from dataFunctions import normalizeFeature
from dataFunctions import centerData

#Python implementation of PCA. Uses numpy, matplotlib, PCA implementation by sklearn,
#and function collection "dataFunctions.py".

#Read and perform normalization on the data
X,names,types = readDataFile("zoo.data")
X_centered = centerData(X)
X_c_and_norm = normalizeFeature(X_centered,12)
X_norm = normalizeFeature(X,12)
X_data = X_norm

#Get the resulting low-dimensional data matrix
pca = PCA(n_components=2)
Y = pca.fit_transform(X_data)
#Y = pca.fit_transform(X)
#Y = pca.fit_transform(X_c_and_norm)

#Plot results
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.scatter(Y[:,0],Y[:,1],c=types)
for i,txt in enumerate(types):
    plt.annotate(str(int(txt)),Y[i])
plt.show()
