import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def rmse(x_gt,x):
    return np.sqrt(np.sum((x_gt-x)**2)/np.size(x))

def gauss(x,mu,variance):
    return np.exp(-((x-mu)**2)/(2*variance))

def gaussMulti(x,mu,covar):
    aux=1/(np.sqrt(((2*np.pi)**2)*np.linalg.det(covar)))
    res=aux*np.exp(-0.5*(((x-mu).conj().T.dot(np.linalg.inv(covar).dot((x-mu))))))
    return res

def polynomialRegression(featuresMax,startFeatures,ridge,train,test,dataset):
    errortest = np.zeros(featuresMax-startFeatures+1)
    errortrain = np.zeros(featuresMax-startFeatures+1)
    errordataset = np.zeros(featuresMax-startFeatures+1)
    for i in range(startFeatures, featuresMax+1):
        features = i
        fMtrain = np.zeros([features, np.int(np.size(train, 0))])
        fMtest = np.zeros([features, np.int(np.size(test, 0))])
        fMset = np.zeros([features, np.int(np.size(dataset, 0))])
        for j in range(0, features):
            fMtrain[j] = train[:, 0] ** j
            fMtest[j] = test[:, 0] ** j
            fMset[j] = dataset[:, 0] ** j

        W = ridge * np.identity(features)
        y = train[:, 1]
        params = np.linalg.pinv((fMtrain.dot(fMtrain.conj().T) + W)).dot(fMtrain.dot(y))

        errortrain[i - startFeatures] = rmse(train[:, 1], fMtrain.conj().T.dot(params))
        errortest[i - startFeatures] = rmse(test[:, 1], fMtest.conj().T.dot(params))
        errordataset[i - startFeatures] = rmse(dataset[:, 1], fMset.conj().T.dot(params))
    #plt.plot(range(startFeatures,featuresMax+1),errortest)
    #Degree 20 with a rmse of .721895
    #plt.plot(range(startFeatures,featuresMax+1),errortrain) #Degree 19 with a rmse of .4611484
    #red_patch = mpatches.Patch(color='blue', label='test')
    #blue_patch = mpatches.Patch(color='green', label='train')
    #plt.legend(handles=[red_patch,blue_patch])
    #plt.show()
    #print(np.argmin(errordataset))#Degree 22 with a rmse of .70722 in case of full dataset trained on the train set
    #print(np.min(errordataset)) #Degree 25 with a rmse of .48887 in case of full dataset trained on the test set
    #plt.plot(dataset[:,1],'ro')
    #plt.plot(fMset.conj().T.dot(params))
    #red_patch = mpatches.Patch(color='red', label='Dataset')
    #blue_patch = mpatches.Patch(color='blue', label='Fitting')
    #plt.legend(handles=[red_patch,blue_patch])
    #plt.show()


def linearRidgeRegression(featuresMax,fMtrain,fMtest,fMset,ridge,train,test):
    errortest = np.zeros(featuresMax - 1)
    errortrain = np.zeros(featuresMax - 1)
    errordataset = np.zeros(featuresMax - 1)
    for i in range(1, featuresMax):
        W = ridge * np.identity(i)
        y = train[:, 1]
        params = np.linalg.pinv((fMtrain.dot(fMtrain.conj().T) + W)).dot(fMtrain.dot(y))
        errortrain[i - 1] = rmse(train[:, 1], fMtrain.conj().T.dot(params))
        errordataset[i - 1] = rmse(dataset[:, 1], fMset.conj().T.dot(params))
        errortest[i - 1] = rmse(test[:, 1], fMtest.conj().T.dot(params))

def polynomialFeatures(featuresMax,train,test,dataset):
    for i in range(1, featuresMax):
        features = i
        fMtrain = np.zeros([features, np.int(np.size(train, 0))])
        fMtest = np.zeros([features, np.int(np.size(test, 0))])
        fMset = np.zeros([features, np.int(np.size(dataset, 0))])
        for j in range(0, features):
            fMtrain[j] = train[:, 0] ** j
            fMtest[j] = test[:, 0] ** j
            fMset[j] = dataset[:, 0] ** j
    return fMtrain,fMtest,fMset


def polynomialWeights(features,train):
    fMtrain = np.zeros([features, np.int(np.size(train, 0))])
    for j in range(0, features):
        fMtrain[j] = train[:, 0] ** j
    W = ridge * np.identity(features)
    return W,fMtrain


def gaussianFeatures(featuresMax,train,test,dataset):
    variance=0.02
    for i in range(1, featuresMax):
        features = i
        fMtrain = np.zeros([features, np.int(np.size(train, 0))])
        fMtest = np.zeros([features, np.int(np.size(test, 0))])
        fMset = np.zeros([features, np.int(np.size(dataset, 0))])
        for j in range(0, features):
            mu=j*2/features
            fMtrain[j] = gauss(train[:,0],mu,variance)
        norm=np.sum(fMtrain,0)
        fMtrain=fMtrain / norm

    return fMtrain

def gaussianRegression(featuresMax,startFeatures,ridge,train,test,dataset):
    variance=0.02
    errortest = np.zeros(featuresMax - startFeatures+1)
    errortrain = np.zeros(featuresMax - startFeatures+1)
    errordataset = np.zeros(featuresMax - startFeatures+1)
    for i in range(startFeatures, featuresMax+1):
        features = i
        fMtrain = np.zeros([features, np.int(np.size(train, 0))])
        fMtest = np.zeros([features, np.int(np.size(test, 0))])
        fMset = np.zeros([features, np.int(np.size(dataset, 0))])
        for j in range(0, features):
            mu = j * 2 / features
            fMtrain[j] = gauss(train[:, 0], mu, variance)
            fMtest[j] = gauss(test[:, 0], mu, variance)
            fMset[j] = gauss(dataset[:, 0], mu, variance)
        normtrain = np.sum(fMtrain, 0)
        normtest = np.sum(fMtest, 0)
        normset = np.sum(fMset, 0)
        fMtrain = fMtrain / normtrain
        fMtest = fMtest / normtest
        fMset = fMset / normset
        W = ridge * np.identity(features)
        y = train[:, 1]
        params = np.linalg.pinv((fMtrain.dot(fMtrain.conj().T) + W)).dot(fMtrain.dot(y))
        errortrain[i - startFeatures] = rmse(train[:, 1], fMtrain.conj().T.dot(params))
        errortest[i - startFeatures] = rmse(test[:, 1], fMtest.conj().T.dot(params))
        errordataset[i - startFeatures] = rmse(dataset[:, 1], fMset.conj().T.dot(params))
    print(np.shape(errortest))
    print(np.argmin(errortest))# 8 Features with a RMSE of 0.82344. But in our 15 to 40 scenario the best is 15 with a rmse of 0.8247, even if our error delta is very small (10^.3)
    print(min(errortest))#
    #plt.plot(range(startFeatures,featuresMax+1),errortest,color='blue')  # Degree 19 with a rmse of .4611484
    #plt.plot(range(startFeatures,featuresMax+1),errortrain,color='red')  # Degree 19 with a rmse of .4611484
    #print(np.shape(errortest))
    #red_patch = mpatches.Patch(color='blue', label='test')
    #blue_patch = mpatches.Patch(color='red', label='train')
    #plt.legend(handles=[red_patch])
    #plt.show()
    #plt.plot(train[:,1],'ro')
    #plt.plot(fMtrain.conj().T.dot(params))
    #red_patch = mpatches.Patch(color='red', label='Dataset')
    #blue_patch = mpatches.Patch(color='blue', label='Fitting')
    #plt.legend(handles=[red_patch,blue_patch])
    #plt.show()
    return errortest

def bayesianRegression(features,points,dataset):
    variance=0.0025
    errortest = np.zeros(np.size(points))
    errortrain = np.zeros(np.size(points))
    errordataset = np.zeros(np.size(points))
    means=np.zeros(np.size(points))
    devs=np.zeros(np.size(points))
    for i in range(0,np.size(points)):
        train=dataset[0:points[i]]
        test=dataset[points[i]:np.size(dataset,0)]
        initWeights,fMtrain=polynomialWeights(features,train)
        beta=((np.size(train,0))/np.sum((train[:,1]-initWeights.dot(fMtrain))**2))#CHECK LATER
        print(beta)
        fMtest = np.zeros([features, np.int(np.size(test, 0))])
        fMset = np.zeros([features, np.int(np.size(dataset, 0))])
        for j in range(0, features):
            fMtest[j] = test[:, 0] ** j
            fMset[j] = dataset[:, 0] ** j
        y = train[:, 1]
        params = np.linalg.pinv((fMtrain.dot(fMtrain.conj().T) +np.identity(features)/(variance*beta))).dot(fMtrain.dot(y))
        variancepost=np.zeros(features)
        variancepost=(1/variance)*np.identity(np.size(features))+beta*fMtrain.dot(fMtrain.conj().T)
        meanpost=beta*(variancepost.dot(fMtrain.dot(train[:,1])))
        predmean=fMtest.conj().T.dot(meanpost)
        #predvariance=np.identity(np.size(test,0))/beta+fMtest.conj().T.dot(variancepost.dot(fMtest))
        predvariance=np.zeros(np.size(test,0))
        for k in range (0,np.size(test,0)):
            predvariance[k]=1/beta+fMtest[:,k].conj().T.dot(variancepost.dot(fMtest[:,k]))
        errortrain[i] = rmse(train[:, 1], fMtrain.conj().T.dot(params))
        errortest[i]= rmse(test[:, 1], fMtest.conj().T.dot(params))
        errordataset[i]= rmse(dataset[:, 1], fMset.conj().T.dot(params))
        print(1/(variance*beta))
        plt.plot(test[:,1],'ro')
        #plt.plot(predvariance, color='red')  # Degree 19 with a rmse of .4611484
        #red_patch = mpatches.Patch(color='red', label='predicted variance')
        blue_patch = mpatches.Patch(color='red', label='predicted variance')
        #plt.legend(handles=[red_patch])
        plt.show()
    print(errortest)
    return errortest

dataset=np.loadtxt("linRegData.txt")
train=np.loadtxt("linRegData.txt")[0:20]
test=np.loadtxt("linRegData.txt")[20:150]
featuresMax=40
startFeatures=15
ridge=10^-6
errortest=np.zeros(featuresMax-1)
errortrain=np.zeros(featuresMax-1)
errordataset=np.zeros(featuresMax-1)

##############################################################################
#PRINT EVERYTHING AGAIN EVENTUALLY BEFORE HANDING IT IN!!!!!!!!!!!!!!!!!!!!!!!!
#gaussianRegression(featuresMax,startFeatures,ridge,train,test,dataset)
#polynomialRegression(featuresMax,startFeatures,ridge,train,test,dataset)
##############################################################################
bayesianRegression(12,[10,12,16,20,50,150],dataset)

