import numpy as np
import scipy as sp
import matplotlib.patches as mpatches
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal as mvn
from numpy.core.umath_tests import matrix_multiply as mm



def kernelGauss(x,y):
    sigma = 1.9
    ker =np.exp(-(np.linalg.norm(x-y,2))/(2*sigma**2))
    return ker


def kernelPoly(x,y):
    c = -2
    d = 3
    return (x.dot(y.conj().T)+c)**d

def lossFunction(alpha,x,y):
    loss=0
    for i in range(0,100):
        for j in range(0,100):
            #loss=loss+alpha[i]*alpha[j]*y[i]*y[j]*kernelGauss(x[i,:],x[j,:])
            loss=loss+alpha[i]*alpha[j]*y[i]*y[j]*kernelPoly(x[i,:],x[j,:])

    return -sum(alpha)+1/2*loss


dataset=np.random.permutation(np.loadtxt("iris-pca.txt"))
#dataset=np.loadtxt("iris-pca.txt")

y=dataset[:,2]
x=dataset[:,0:2]
bounds=np.zeros([100,2])
bounds[:,1]=2
const=({'type':'eq','fun' : lambda alpha : np.sum(alpha*y)})
        #{'type': 'ineq', 'fun': lambda alpha: alpha},
        #{'type':'ineq','fun' : lambda alpha : alpha-0*np.ones(np.size(alpha))})
results=optimize.minimize(lossFunction,np.zeros(100),args=(x,y),bounds=bounds,constraints=const)
print(results.x)
print(np.nonzero(results.x))
