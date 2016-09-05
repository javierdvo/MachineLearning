import numpy as np
import matplotlib.patches as mpatches
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal as mvn
from numpy.core.umath_tests import matrix_multiply as mm
from sklearn import metrics as met

def  relu(x):
    return x * (x > 0)

def lossfunction(x,y):
    return met.log_loss(x,y)

def backprop(batchIn,batchOut,learnRate,hiddenUnits,features,batchSize):
    W1 = np.ones([hiddenUnits, features])*0.01
    W2 = np.ones(hiddenUnits)*0.01
    a = np.zeros(hiddenUnits)
    z = np.zeros(hiddenUnits)
    for t in range(0, 100):
        batcherror2 = np.zeros(hiddenUnits)
        batcherror1 = np.zeros([hiddenUnits, features])
        error1 = np.zeros([hiddenUnits, features])
        for n in range(0, batchSize-1):
            for j in range(0, hiddenUnits):
                a[j] = (W1[j, :].dot(batchIn[n, :]))
                z[j] = relu(a[j])
            ak = W2.dot(z)
            deltakn = ak - batchOut[n]
            deltajn = W2 * deltakn
            error2 = deltakn * z
            for k in range(0, hiddenUnits):
                error1[k, :] = deltajn[k] * a[k] * batchIn[n, :]
            batcherror2 = batcherror2 + error2
            batcherror1 = batcherror1 + error1
        W2 = W2 - learnRate * batcherror2
        W1 = W1 - learnRate * batcherror1
        print(a)
        print(deltajn)
    return W1,W2

features=784
hiddenUnits= int(np.round(1.2*features))
learnRate=0.07
batchSize=500
test_in=np.loadtxt("mnist_small_test_in.txt",delimiter=',')
test_out=np.loadtxt("mnist_small_test_out.txt",delimiter=',')
train_in=np.loadtxt("mnist_small_train_in.txt",delimiter=',')
train_out=np.loadtxt("mnist_small_train_out.txt",delimiter=',')

#for i in range(0,10):
indexes=np.random.permutation(range(0,6006))
batchIn=train_in[indexes[0:batchSize],:]
batchOut=train_out[indexes[0:batchSize]]
w1,w2=backprop(batchIn,batchOut,learnRate,hiddenUnits,features,batchSize)
output=np.zeros(np.size(train_in,1))
for n in range(0,np.size(train_in,1)):
    output[n]=w2.dot(relu(w1.dot(train_in[n])))
print(output)
print(lossfunction(train_out,output))








