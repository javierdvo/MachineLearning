#from numpy import stats
import numpy as np
import matplotlib.patches as mpatches
from scipy import stats
import matplotlib.pyplot as plt

def plothist(size,data1,data2,data3):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind1 = data1[1]
    ind2= data2[1]
    ind3 = data3[1]
    width = size
    rects1 = ax.bar(ind1, data1[0], width*100,
                    color='blue', label='2.0  h')
    plt.legend(handles=[rects1])
    plt.show()
    rects2 =ax.bar(ind2, data2[0], width*25,
           color='red', label='0.5 h')
    plt.legend(handles=[rects2])
    plt.show()

    rects3 =ax.bar(ind3, data3[0], width,
           color='green', label='0.02 h')
    plt.legend(handles=[rects3])

    plt.show()

def kernel(u,h,d):
    kernel = (1 / ((h * np.sqrt(2 * np.pi)**d)) * np.exp(-((np.abs(u))**2) / (2*h*h)))
    return kernel

def KDE(h,d,data):
    x = np.arange(-4, 8 + h, h)
    kX = np.zeros(np.size(x))
    for i in range(0, np.size(x)):
        kX[i] = np.sum(kernel(x[i] - data, h, d))
    prob = kX / np.size(data)
    return prob,x

def kNearest(size,data,point):
    z = np.sort(data)
    for i in range (0,size-1):
        coord = np.searchsorted(z, point)
        if coord==0 :
            z = np.delete(z, coord)
        elif coord==np.size(z):
            z = np.delete(z, coord-1)
        elif(point-z[coord-1])>(z[coord]-point):
            z = np.delete(z, coord)
        else:
            z = np.delete(z, coord-1)
    coord = np.searchsorted(z, point)
    if coord == np.size(z):
        return np.abs(point-z[coord-1])
    elif coord==0:
        return np.abs(point - z[coord])
    elif(point-z[coord-1])>(z[coord]-point):
        return np.abs(point - z[coord])
    else:
        return np.abs(point - z[coord-1])

def kNN(kSize,data):
    x = np.arange(-4, 8, 0.05)
    kN = np.zeros(np.size(x))
    for i in range(0, np.size(x)):
        kN[i] = kSize / (kNearest(kSize, data, x[i]) * np.size(data))

    return kN,x

def histogram(size,data):
    bins = np.arange(int(np.floor(min(data))),int(np.ceil(max(data))),size)
    res=np.zeros((2,np.size(bins)))
    res[1,:]=bins
    z=0
    for i in range(0,np.size(bins)):
        for x in range (0,np.size(data),1):
            if data[x]>bins[i] and data[x]<=(bins[i]+size):
                res[0,i]+=1
    print(np.sum(res[0,:]/np.size(data)))
    res[0]=res[0]/np.size(data)
    return res



train=np.loadtxt("nonParamTrain.txt")
test=np.loadtxt("nonParamTest.txt")

print(np.shape(test))#Histogram


#A bit more thinking
size=2
a=histogram(size,train)
size=.5
b=histogram(size,train)
size=.02
c=histogram(size,train)
#plothist(size,a,b,c)



#Kernel Density Estimation

h=0.8
d=1
prob1,x1=KDE(h,d,train)
h=0.2
d=1
prob2,x2=KDE(h,d,train)
h=0.03
d=1
prob3,x3=KDE(h,d,train)
print(np.sum(np.log(prob1)))
print(np.sum(np.log(prob2)))
print(np.sum(np.log(prob3[(np.nonzero(prob3))])))
#plota=plt.plot(x1,prob1,color='blue', label='0.8')
#plotb=plt.plot(x2,prob2,color='red', label='0.2')
#plotc=plt.plot(x3,prob3,color='green', label='0.03')

#red_patch = mpatches.Patch(color='blue', label='0.8')
#blue_patch = mpatches.Patch(color='red', label='0.2')
#green_patch = mpatches.Patch(color='green', label='0.03')
#plt.legend(handles=[red_patch,blue_patch,green_patch])

#plt.show()


#K Nearest Neighbours
kSize=2
kN1,x1=kNN(kSize,train)
kSize=8
kN2,x2=kNN(kSize,train)
kSize=35
kN3,x3=kNN(kSize,train)
print(np.sum(np.log(kN1)))
print(np.sum(np.log(kN2)))
print(np.sum(np.log(kN3)))
#plt.plot(x1,kN1,color='green')
#plt.plot(x2,kN2,color='red')
#plt.plot(x3,kN3,color='blue')
#red_patch = mpatches.Patch(color='green', label='2')
#blue_patch = mpatches.Patch(color='red', label='8')
#green_patch = mpatches.Patch(color='blue', label='35')
#plt.legend(handles=[red_patch,blue_patch,green_patch])
#plt.show()

print("----------------------------------------------")
#test set:
h=0.8
d=1
prob1,x1=KDE(h,d,test)
stats.norm
h=0.2
d=1
prob2,x2=KDE(h,d,test)
h=0.03
d=1
prob3,x3=KDE(h,d,test)
print(np.sum(np.log(prob1)))
print(np.sum(np.log(prob2)))
print(np.sum(np.log(prob3[(np.nonzero(prob3))])))

kSize=2
kN1,x1=kNN(kSize,test)
kSize=8
kN2,x2=kNN(kSize,test)
kSize=35
kN3,x3=kNN(kSize,test)
print(np.sum(np.log(kN1)))
print(np.sum(np.log(kN2)))
print(np.sum(np.log(kN3)))



