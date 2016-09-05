import numpy
import matplotlib.pyplot as plt
st0=numpy.array
st1=numpy.array
resultsSample=numpy.array
st0=[0.015, 0.985]
st1=[[0.35, 0.65],
     [0.015, 0.985]]
ct0=[0.015, 0.985]
ct1=[[0, 1],
     [0.015, 0.985]]

resultsSample=[0.015]
resultsCatastrophe=[0.015]
for i in range(15):
    st0=numpy.dot(st0, st1)
    st1[1] = list(st0.copy())
    resultsSample.append(st0[0])
    ct0=numpy.dot(ct0, ct1)
    #resultsCatastrophe.append(ct0[0])
    resultsCatastrophe.append(ct0[0]+resultsCatastrophe[i])
    ct1[1] = list(ct0.copy())

#x = numpy.linspace(0, 15, 15)
plt.plot(range(16),resultsSample, label='Sample')
plt.plot(range(16), resultsCatastrophe, label='Catastrophe')
#plt.plot(x, x**3, label='cubic')

plt.xlabel('Generations')
plt.ylabel('Probability')

plt.title("Umbrella Corp.")

plt.legend()

plt.show()
