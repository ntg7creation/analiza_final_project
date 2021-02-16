"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random
class Point(object):
    def __init__(self):
        """ Create a new point at the origin """
        self.x:float = 0.0
        self.y:float = 0.0
    pass


def get_lambda( exponent: int, array: list) -> callable:
    if exponent == -1:
        return lambda x: 0;
    temp = lambda x: get_lambda((exponent - 1), array)(x) + array[exponent] * (x ** exponent)
    return temp


class lest_squre:

    def get_lambda(self,exponent: int, array: list) -> callable:
        if exponent == -1:
            return lambda x: 0;
        temp = lambda x: self.get_lambda((exponent - 1), array)(x) + array[exponent] * (x ** exponent)
        return temp
        pass

    def lest_squre_curve( pointsx:list,pointsy:list,n:int, d:int, )->callable:


        k = d
        coefficient_matrix = np.zeros(shape=(k + 1, k + 1))
        # make first line
        for i in range(0, k + 1):
            # clac sum
            sum = 0.0;
            for j in range(n):
                sum += pointsx[j] ** i
            coefficient_matrix[0, i] = sum
        # calc matrix
        for i in range(1, k + 1):
            sum = 0.0;
            for j in range(n):
                sum += pointsx[j] ** (k + i)
            temparray = np.append(coefficient_matrix[i - 1][1:], sum)
            # temparray.append(sum)
            coefficient_matrix[i] = temparray

        # calc result vector
        coefficient_result = [0] * (k + 1)
        for i in range(0, k + 1):
            # clac sum
            sum = 0.0;
            for j in range(n):
                sum += (pointsx[j] ** i) * pointsy[j]
            coefficient_result[i] = sum

        # calc determinet
        detM = np.linalg.det(coefficient_matrix)
        Marray = []
        detarray = []
        Avalues = []
        for i in range(k + 1):
            temp = np.transpose(coefficient_matrix.copy())
            temp[i] = coefficient_result
            Marray.append(np.transpose(temp))
            detarray.append(np.linalg.det(Marray[i]))
            Avalues.append(detarray[i] / detM)

        def get_lambda(exponent: int, array: list) -> callable:
            if exponent == -1:
                return lambda x: 0;
            temp = lambda x: get_lambda((exponent - 1), array)(x) + array[exponent] * (x ** exponent)
            return temp
            pass

        function = get_lambda(k, Avalues)
        return function

time_for_nx6 =[0.0]*13;
class Assignment4A:
    def __init__(self):

        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        for exponent in range (1,13):
            k = exponent
            n = 6
            pointlistx = [-3, -2, -1, -0.2, 1, 3]  # size n
            pointlisty = [0.9, 0.8, 0.4, 0.2, 0.1, 0]
            start = time.time()
            lest_squre.lest_squre_curve(pointlistx,pointlisty,n,k)
            end = time.time()
            time_for_nx6[exponent] = (end - start)

        pass

    # def get_lambda(self,exponent: int, array: list) -> callable:
    #     if exponent == -1:
    #         return lambda x: 0;
    #     temp = lambda x: self.get_lambda((exponent - 1), array)(x) + array[exponent] * (x ** exponent)
    #     return temp
    #     pass

    def lest_squre_curve(self, f: callable, a: float, b: float, d:int, maxtime: float)->callable:
        starttime = maxtime
        start = time.time()
        # print("time")
        # print(start)
        n=0
        time_for_6_points=0
        if d < 13 & d>=0 :
            time_for_6_points = time_for_nx6[d]
        if time_for_6_points == 0:
            n =1500
        else:
            n = 3 * maxtime/time_for_nx6
        if n > 1500:
            n = 1500
        counter = 0
        pointlist = [Point] * n;
        for i in range(0, n):
            now = time.time()
            pointlist[i].x = a + (abs(a - b) / n) * i
            pointlist[i].y = f(pointlist[i].x)
            counter += 1
            if now - start > maxtime/4:
                n = counter
                # print(now)
                break;
        if n <= 1:
            return lambda x: pointlist[0].y
        k = d
        coefficient_matrix = np.zeros(shape=(k + 1, k + 1))
        # make first line
        for i in range(0, k + 1):
            # clac sum
            sum = 0.0;
            for j in range(n):
                sum += pointlist[j].x ** i
            coefficient_matrix[0, i] = sum
        #calc matrix
        for i in range(1, k + 1):
            sum = 0.0;
            for j in range(n):
                sum += pointlist[j].x ** (k + i)
            temparray = np.append(coefficient_matrix[i - 1][1:], sum)
            # temparray.append(sum)
            coefficient_matrix[i] = temparray

        #calc result vector
        coefficient_result = [0] * (k + 1)
        for i in range(0, k + 1):
            # clac sum
            sum = 0.0;
            for j in range(n):
                sum += (pointlist[j].x ** i) * pointlist[j].y
            coefficient_result[i] = sum

        # calc determinet
        # print(coefficient_matrix)
        detM = np.linalg.det(coefficient_matrix)
        Marray = []
        detarray = []
        Avalues = []
        for i in range(k + 1):
            temp = np.transpose(coefficient_matrix.copy())
            temp[i] = coefficient_result
            Marray.append(np.transpose(temp))
            detarray.append(np.linalg.det(Marray[i]))
            if detM == 0 :
                return lambda x:0; # the numbers we are dealing with are too small
            # print("det")
            # print(detM)
            # print(detarray[i])
            Avalues.append(detarray[i] / detM)

        def get_lambda(exponent: int, array: list) -> callable:
            if exponent == -1:
                return lambda x: 0;
            temp = lambda x: get_lambda((exponent - 1), array)(x) + array[exponent] * (x ** exponent)
            return temp
            pass

        function = get_lambda(k, Avalues)
        return  function

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        # replace these lines with your solution
        result = self.lest_squre_curve(f, a, b, d, maxtime)
        # y = f(1)

        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):            
            self.assertNotEquals(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)

        
        



if __name__ == "__main__":
    unittest.main()
