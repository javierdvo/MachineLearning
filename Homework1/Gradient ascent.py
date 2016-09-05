import numpy

def rosenbock(x):
    return sum(100.0*(x[:1]-x[1:]**2.0)**2.0 + (x[1:]-1)**2.0)


def rosenbockX(x,y):
    return -400*(y-x**2)*x-2*(1-x)

def rosenbockY(x,y):
    return 200*(y-x**2)

iterations=5
learnRate=0.2
pointx=1
pointy=2
switch=0
while iterations>0:
    print(pointx)
    print(pointy)
    pointxold=pointx
    pointx=pointxold-learnRate*rosenbockX(pointxold,pointy)
    pointy=pointy-learnRate*rosenbockY(pointxold,pointy)
    iterations-=1
