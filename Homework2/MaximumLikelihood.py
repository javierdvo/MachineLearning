#from numpy import stats
import numpy as np
import matplotlib.patches as mpatches
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.mlab as mlab




def gaussMulti(x,mu,covar):

    aux=1/(np.sqrt(((2*np.pi)**2)*np.linalg.det(covar)))
    res=aux*np.exp(-0.5*(((x-mu).conj().T.dot(np.linalg.inv(covar).dot((x-mu))))))
    return res

class1=np.loadtxt("densEstClass1.txt")
class2=np.loadtxt("densEstClass2.txt")
class1M=np.asmatrix(class1)
class2M=np.asmatrix(class2)

size1=np.float64(np.size(class1,0))
size2=np.float64(np.size(class2,0))
N=(size1+size2)
prior1=size1/N
prior2=size2/N

mean1=[(1/size1)*np.sum(class1M[:,0]),(1/size1)*np.sum(class1M[:,1])]
mean2=[(1/size2)*np.sum(class2M[:,0]),(1/size2)*np.sum(class2M[:,1])]

covariance1 = np.zeros([2, 2])
covariance2 = np.zeros([2, 2])
print(size1)
for k in np.arange(0,int(size1)-1):
    covariance1 = covariance1 + np.outer(class1M[k] - mean1, class1M[k] - mean1)
for k in np.arange(0,int(size2)-1):
    covariance2 = covariance2 + np.outer(class2M[k] - mean2, class2M[k] - mean2)
covariance1 = covariance1* (1 / size1)
covariance2 = covariance2* (1 / size2)

uncovariance1 = covariance1* (1 / (size1-1))
uncovariance2 = covariance2* (1 / (size2-1))

pdf1=np.zeros([size1,1])
pdf2=np.zeros([size2,1])
for k in np.arange(0, size1-1):
    pdf1[k]=gaussMulti(class1[k],mean1,uncovariance1)
for k in np.arange(0, size2 - 1):
    pdf2[k] = gaussMulti(class2[k], mean2, uncovariance2)


#X, Y = np.meshgrid(class1[0:size1,0],class1[0:size1,1])
#Z1 = mlab.bivariate_normal(X, Y, covariance1[0,0], covariance1[1,1], mean1[0], mean1[1],covariance1[1,0])
#plt.contour(X,Y,Z1,linewidths=0.1,zorder=-1,inline=1)
#X, Y = np.meshgrid(class2[0:size2,0],class2[0:size2,1])
#Z2 = mlab.bivariate_normal(X, Y, covariance2[0,0], covariance2[1,1], mean2[0], mean2[1],covariance2[1,0])
#plt.contour(X,Y,Z2,linewidths=0.1,zorder=-1,inline=1)
#plt.scatter(class1M[:,0],class1M[:,1],20,color='red',zorder=1,)
#plt.scatter(class2M[:,0],class2M[:,1],20,color='blue',zorder=1)
#plt.show()

#ll_new = 0
#for mu, sigma in zip(mean1, covariance1):
    # ll_new += pi * mvn(mu, sigma).pdf(xs)
#    ll_new += mvn(mu, sigma).pdf(class1)
#logLike[i] = np.log(ll_new).sum()

posterior1=np.log(pdf1)*prior1/prior2
posterior2=np.log(pdf2)*prior2/prior1
plt.plot(posterior1)
plt.plot(posterior2)
plt.show()


#class1M=np.asmatrix(class1)
#class2M=np.asmatrix(class2)
#plt.axis([-10, 11, -10, 11])
#plt.scatter(class1M[:,0],class1M[:,1],30,'red')
#plt.scatter(class2M[:,0],class2M[:,1],30,'blue')
#plt.contour(likelihood1)
#plt.show()#

#posterior1=np.asmatrix(likelihood1*prior1)
#posterior2=np.asmatrix(likelihood2*prior2)
#plt.plot(posterior1[:,0],posterior1[:,1])
#plt.plot(posterior2[:,0],posterior2[:,1])
#plt.show()


#priorMean=gauss(0,mean1,variance1)