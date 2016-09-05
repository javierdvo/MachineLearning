import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

def normalize(dataset):
    normdataset=np.copy(dataset)
    for i in range(0,np.size(dataset,1)):
        mean=np.mean(dataset[:,i])
        normdataset[:,i]=dataset[:,i]-mean
        normdataset[:,i]=normdataset[:,i]/np.std(normdataset[:,i])
    return normdataset



def computepca(dataset):
    U, s, V = np.linalg.svd(dataset,full_matrices=True)
    eigVecs,eigvals=np.linalg.eig(np.cov(dataset))
    S=sp.svdvals(dataset)
    print(S)
    lambda1 = S ** 2 / np.size(dataset, 1)
    cumvar = np.cumsum(lambda1) / np.sum(lambda1)
    return U,lambda1,cumvar

def rmse(x_gt,x):
    return np.sqrt(np.sum((x_gt-x)**2)/np.size(x))

def projection(data,U,n):
    proj=np.zeros([np.size(data,0),n])
    for i in range(0,np.size(data,1)):
        proj[i,:]=U[:,0:n].conj().T.dot(data[:,i])
    return proj

def scatterplot(data,proj):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(0,np.size(data,0)):
        if(data[i,4]==1):
            ax.scatter(proj[i,0],proj[i,1],c='yellow')
        elif(data[i,4]==2):
            ax.scatter(proj[i,0],proj[i,1],c='red')
        else:
            ax.scatter(proj[i,0],proj[i,1],c='black')
    plt.show()



dataset=np.loadtxt("iris.txt",delimiter=',')
normset=normalize(dataset[:,0:4])
Um,lamda,cumulative=computepca(normset)
plt.plot(range(1,5),cumulative)
plt.show()
print(lamda.shape)
print(cumulative.shape)
proj=projection(normset,Um,np.searchsorted(cumulative,0.95)+1)
print(proj)
scatterplot(dataset,proj)
