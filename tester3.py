import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

class Point(object):
    def __init__(self):
        """ Create a new point at the origin """
        self.x:float = 0.0
        self.y:float = 0.0
    pass

k = 12
n =6
pointlist = [-3,-2,-1,-0.2,1,3] # size n
pointlisty =[0.9,0.8,0.4,0.2,0.1,0]
coefficient_matrix = np.zeros(shape= (k+1,k+1))
# make first line
for i in range(0, k+1):
    # clac sum
    sum = 0.0;
    for j in range(n):
        sum += pointlist[j] ** i
    coefficient_matrix[0, i] = sum
for i in range(1,k+1):

    sum = 0.0;
    for j in range(n):
        sum += pointlist[j] ** (k+i)
    temparray = np.append(coefficient_matrix[i - 1][1:],sum)
    # temparray.append(sum)
    coefficient_matrix[i] = temparray


coefficient_result = [0]*(k+1)
for i in range(0, k+1):
    # clac sum
    sum = 0.0;
    for j in range(n):
        sum +=( pointlist[j] ** i) *pointlisty[j]
    coefficient_result[i] = sum

#calc determinet
print(coefficient_matrix)
print(coefficient_result)
print()
print()
detM = np.linalg.det(coefficient_matrix)
Marray = []
detarray = []
Avalues=[]
for i in range (k+1) :
    temp = np.transpose(coefficient_matrix.copy())
    temp[i] = coefficient_result
    # print("temp:")
    # print(temp)
    # print()
    Marray.append(np.transpose(temp))
    detarray.append(np.linalg.det(Marray[i]))
    Avalues.append(detarray[i]/detM)
# print(Marray)
# print(detarray)
print(Avalues)




def get_lambda(exponent: int, array: list) -> callable:
    if exponent == -1:
        return lambda x: 0;
    temp = lambda x: get_lambda((exponent - 1), array)(x) + array[exponent] * (x ** exponent)
    return temp
    pass


function = get_lambda(k,Avalues)


a=-10
b=10
n=40
arrayx = [0]*n
arrayy = [0]*n
for i in range(0, n):
    arrayx[i] = a + (abs(a - b) / n) * i
    arrayy[i] = function(arrayx[i])
print(arrayx)
print(arrayy)

plt.plot(arrayx, arrayy, 'o', color='black');
plt.show()