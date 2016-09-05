print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
def kernelPoly(x,y):
    c = -2
    d = 3
    return (x.dot(y.conj().T)+c)**d

def kernelGauss(x,y):
    sigma = 1.9
    ker =np.exp(-(np.linalg.norm(x-y)**2)/(2*sigma**2))
    return ker

dataset=np.loadtxt("iris-pca.txt")

y=dataset[:, 2]-1
X=dataset[:, 0:2]


#iris = datasets.load_iris()
#print(iris)
#X = iris.data
#y = iris.target

#X = X[y != 1, :2]
#y = y[y != 1]
n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.float)

X_train = X
y_train = y


# fit the model
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(C=2,kernel=kernelPoly, gamma=10)
    clf.fit(X_train, y_train)
    print(np.size(clf.dual_coef_))
    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)



    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()