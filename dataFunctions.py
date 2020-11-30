import numpy as np

#Collection of functions used for classical MDS and Isomap. Uses numpy.

#Read from data file structured according to https://archive.ics.uci.edu/ml/datasets/zoo
def readDataFile(inputString):
    f = open(inputString,"r")
    lines = f.readlines()

    N=len(lines)
    K=16
    X=np.zeros([N,K])
    names = ["" for i in range(N)]
    types= np.zeros(N)

    for i,line in enumerate(lines):
        tempLine = line.strip()
        tempLine = tempLine.split(",")
        names[i]=tempLine[0]
        types[i]=tempLine[17]
        X[i]=tempLine[1:17]

    return X,names,types

#Normalize specified feature of the data samples to approximately be in range(0,1)
def normalizeFeature(X,col):
    colList = np.copy(X[:,col])
    mean = np.mean(colList)
    var = np.var(colList)
    normalizedCol= (colList)/var

    X_normalized = np.copy(X)
    X_normalized[:,col] = normalizedCol
    return X_normalized

#Shift data to be centered around 0
def centerData(X):
    Y = X-np.mean(X,axis=0)
    return Y

#Euclidian distance between all data samples
def getDistanceMatrix(X):
    N=len(X)
    D = np.zeros([N,N])
    for i in range(len(X)):
        for j in range(len(X)):
            D[i,j]=np.linalg.norm(np.subtract(X[i],X[j]))
    return D

#Double centering trick to retrieve (an approximative) similarity matrix S from distance matrix D
def doubleCentering(D):
    N = len(D)
    J_n=np.ones(np.shape(D))
    D = np.square(D)
    S_temp = -(1/2)*(D-(1/N)*np.dot(D,J_n)-(1/N)*np.dot(J_n,D) + (1/N**2)*(np.dot(J_n,np.dot(D,J_n))))
    S=S_temp
    return S

#Calculate similarity matrix S from the data samples X
def getSimilarityMatrix(X):
    N=len(X)
    S = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            S[i,j]=np.dot(X[i],X[j])
    return S

#Classical multidimensional scaling implementation
def MDS(S,k):
    Eigs, U = np.linalg.eig(S)
    U=np.real(U)
    indicies = np.argsort(Eigs)[::-1]
    Eigs = Eigs[indicies]
    U=U[:,indicies] 

    Sigma = np.diag(Eigs).round(7)
    Sigma = np.abs(np.real(Sigma))
    #S_prim = np.dot(U,np.dot(Sigma,U.transpose()))
    

    I_k_n = np.zeros([k,len(S)])
    for i in range (k):
        I_k_n[i][i]=1
    
    SigmaSqrt = np.sqrt(Sigma).round(7)
    X = np.dot(I_k_n,np.dot(SigmaSqrt,U.transpose()))
    
    return X,Eigs

#Constructs a graph using the matrix of euclidian distances "delta",
#each data point is a vertex with the p nearest data points as edges.
def getGraph(delta,p):
    G_idx = np.zeros([len(delta),p],dtype=int)
    G = np.zeros([len(delta),p])

    tempList=np.zeros(p)
    for i,v in enumerate(delta):
        tempList=np.argsort(v)[0:p]
        G_idx[i,:] = tempList
        G[i,:] = v[tempList]

    N=len(G)
    dist = np.ones([N,N])*np.inf
    for i in range(N):
        dist[i,G_idx[i]]=G[i]
    return dist






