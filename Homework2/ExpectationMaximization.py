#from numpy import stats
import numpy as np
import matplotlib.patches as mpatches
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal as mvn
from numpy.core.umath_tests import matrix_multiply as mm



def em_gmm_vect(xs, pis, mus, sigmas, tol=0.01, max_iter=100):

    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for i in range(max_iter):
        exp_A = []
        exp_B = []
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j in range(k):
            ws[j, :] = pis[j] * mvn(mus[j], sigmas[j]).pdf(xs)
        ws /= ws.sum(0)
        print(ws[1][1])
        # M-step
        pis = ws.sum(axis=1)
        pis /= n

        mus = np.dot(ws, xs)
        mus /= ws.sum(1)[:, None]

        sigmas = np.zeros((k, p, p))
        for j in range(k):
            ys = xs - mus[j, :]
            sigmas[j] = (ws[j,:,None,None] * mm(ys[:,:,None], ys[:,None,:])).sum(axis=0)
        sigmas /= ws.sum(axis=1)[:,None,None]

        # update complete log likelihoood
        ll_new = 0
        for pi, mu, sigma in zip(pis, mus, sigmas):
            ll_new += pi*mvn(mu, sigma).pdf(xs)
        ll_new = np.log(ll_new).sum()

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    return ll_new, pis, mus, sigmas


def gauss(x,mu,variance):
    return (1/(np.sqrt(2*np.pi*variance)))*np.exp(-((x-mu)**2)/(2*variance))

def gaussMulti(x,mu,covar):
    aux=1/(np.sqrt(((2*np.pi)**2)*np.linalg.det(covar)))
    res=aux*np.exp(-0.5*(((x-mu).conj().T.dot(np.linalg.inv(covar).dot((x-mu))))))
    return res

gMM=np.loadtxt("gmm.txt")
iterations=30
modes=4
#means=[[1.4,1.6],[1.5,1.7],[1.3,1.5],[1.4,1.7]]
means=[np.mean(gMM,0)/2,np.mean(gMM,0),np.mean(gMM,0)*1.5,np.mean(gMM,0)*2]
#means=[[1.1,1.1],[1.1,1.1],[1.1,1.1],[1.1,1.1]]
#weights=[[2,3],[1,1.5],[2,0.5],[0.5,2]]
weights=[0.20,0.30,0.15,0.35]
covariances=[np.cov(np.matrix(gMM),rowvar=0),
             np.cov(np.matrix(gMM), rowvar=0),
             np.cov(np.matrix(gMM), rowvar=0),
             np.cov(np.matrix(gMM), rowvar=0)]
#covariances=np.random.rand(4,2,2)
#covariances=[[[1.7,1.8],[1.5,1]],
 #            [[1.8,1.5],[1,1.8]],#
#             [[1.1,1.4],[1.9,1.5]],
#             [[1.4,1.3],[1.5,1.5]]]
resp=np.array(np.zeros([500,modes]))
covOT=np.zeros([30,4,2,2])
logLike=np.zeros(30)
#resp=np.zeros([modes,2,2])
#covariances=[np.cov(gMM),np.cov(gMM),np.cov(gMM),np.cov(gMM)]
covOT[0]=covariances
print('----------------')
print(covariances)
for i in range(0,iterations):
    den=0
    resp = np.zeros((len(weights), np.size(gMM,0)))
    for j in range(0,modes):
        for z in range(0,np.size(gMM,0)):
            #resp[j, i] = weights[j] * mvn(means[j], covariances[j]).pdf(gMM[i])
            resp[j, z] = weights[j] * gaussMulti(gMM[z],means[j], covariances[j])
    resp = resp/resp.sum(0)
    resp=resp.conj().T

    Nj=np.squeeze(np.asarray(np.sum(resp,axis=0)))
    means=np.zeros([4,2])
    covariances=np.zeros([4,2,2])
    for j in range (0,modes):
        for k in range (0,499):
            means[j]=means[j]+(resp[k,j]*gMM[k])
        means[j]=means[j]*(1/Nj[j])
    weights=(Nj/np.size(gMM,0))
    for j in range(0,modes):
        for k in range(0, 499):
            covariances[j] = covariances[j] + (resp[k,j]*np.outer(gMM[k]-means[j],gMM[k]-means[j]))
        covariances[j]=covariances[j]*(1/Nj[j])
    #loglik
    print(covariances)
    covOT[i]=covariances

    ll_new = 0
    for pi, mu, sigma in zip(weights, means, covariances):
        #ll_new += pi * mvn(mu, sigma).pdf(xs)
        ll_new+=pi * mvn(mu, sigma).pdf(gMM)
    logLike[i] = np.log(ll_new).sum()


    if i==33:
        X, Y = np.meshgrid(gMM[0:np.size(gMM, 0), 0], gMM[0:np.size(gMM, 0), 1])
        Z1 = mlab.bivariate_normal(X, Y, covariances[0][0][0], covariances[0][1][1], means[0][0], means[0][1],
                                   covariances[0][1][0])
        Z2 = mlab.bivariate_normal(X, Y, covariances[1][0][0], covariances[1][1][1], means[1][0], means[1][1],
                                   covariances[1][1][0])
        Z3 = mlab.bivariate_normal(X, Y, covariances[2][0][0], covariances[2][1][1], means[2][0], means[2][1],
                                   covariances[2][1][0])
        Z4 = mlab.bivariate_normal(X, Y, covariances[3][0][0], covariances[3][1][1], means[3][0], means[3][1],
                                   covariances[3][1][0])
        plt.contour(X, Y, Z1,6 ,linewidths=0.1, zorder=-1)
        plt.contour(X, Y, Z2,6 ,linewidths=0.1, zorder=-1)
        plt.contour(X, Y, Z3,6 ,linewidths=0.1, zorder=-1)
        plt.contour(X, Y, Z4,6, linewidths=0.1, zorder=-1)
        plt.scatter(gMM[0:np.size(gMM, 0), 0], gMM[0:np.size(gMM, 0), 1], 20, color='red', zorder=1)
        plt.show()
plt.plot(logLike)
plt.show()
#np.savetxt('text1.txt',covOT,fmt='%.64f')