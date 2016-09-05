import numpy


def rosenbock(x):
    return sum(100.0 * (x[:1] - x[1:] ** 2.0) ** 2.0 + (x[1:] - 1) ** 2.0)


def rosenbockx(x, y):
    return -400 * (y - x ** 2) * x - 2 * (1 - x)


def rosenbocky(x, y):
    return 200 * (y - x ** 2)

iterations = 100
learnRate = 0.000861
pointx = 2.1
pointy = -1.7
while iterations > 0:
    print("Iteration %i Point x: %d Point y: %d" % (100-iterations, pointx, pointy))
    pointxold = pointx
    pointx -= learnRate * rosenbockx(pointxold, pointy)
    pointy -= learnRate * rosenbocky(pointxold, pointy)
    iterations -= 1
