"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import random

from functionUtils import AbstractShape


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self):
        pass


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass


    def areabetween2(self, f1: list, f2: list, n: int) -> np.float32:
        # assuming qustion 2 is correct
        first = f1[0][0]
        last = f1[n-1][0] #[-1][0]

        # dx = (abs(first - last) / (n-2))
        areasum = np.float32(0.0)
        # il add minvalue to all ypoints
        for i in range(1, n):
            dx=f1[i][0] - f1[i-1][0]
            prehigh = f1[i-1][1] - f2[i-1][1]
            High = f1[i][1] - f2[i][1]
            # if (np.sign(base1) != np.sign(base2)):
            areasum += (abs(prehigh)  + abs(High)) * dx/2

            #areasum += abs(High) * dx
        return areasum

        pass

    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        n = 21
        matrixa = contour(n)
        half =(int)(n/2)
        firstpart= matrixa[half:]
        secondpart= matrixa[:half+1]
        fliped = np.copy(secondpart)
        for i in range(0,half+1):
            fliped[i] = (secondpart[half - i])
        # print(firstpart)
        # print(fliped)
        resulta = self.areabetween2(fliped,firstpart,half+1)
        print("area1:")
        print(resulta)

        n = 33
        matrixa = contour(n)
        half =(int)(n/2)
        firstpart= matrixa[half:]
        secondpart= matrixa[:half+1]
        fliped = np.copy(secondpart)
        for i in range(0,half+1):
            fliped[i] = (secondpart[half - i])
        resultb = self.areabetween2(fliped,firstpart,half+1)
        print("area2:")
        print(resultb)

        dn = 33 - 21
        derr = abs(resulta-resultb)/maxerr
        n = n+ (int) (4*dn*derr)
        isodd =  + n%2 #make sure n is odd
        n = n+1 +isodd
        matrixa = contour(n)
        half =(int)(n/2)
        firstpart= matrixa[half:]
        secondpart= matrixa[:half+1]
        fliped = np.copy(secondpart)
        for i in range(0,half+1):
            fliped[i] = (secondpart[half - i])

        resultc = self.areabetween2(fliped,firstpart,half+1)
        print("area3:")
        print(resultc)



        matrixa = np.transpose(matrixa)

        xarray = matrixa[0]
        yarray = matrixa[1]
        plt.plot(xarray, yarray, 'o', color='black');
        plt.show()
        # print(xarray)
        # print(yarray)
        # print (contour(13))
        # array = []
        # for i in range (13):
        #     array.append(contour(i))
        # print(array)
        print()
        print()
        print()
        return np.float32(resulta)
    
    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        # replace these lines with your solution
        result = MyShape()
        x, y = sample()

        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #
    #     def sample():
    #         time.sleep(7)
    #         return circ()
    #
    #     ass4 = Assignment4()
    #     T = time.time()
    #     shape = ass4.fit_shape(sample=sample, maxtime=5)
    #     T = time.time() - T
    #     self.assertTrue(isinstance(shape, AbstractShape))
    #     self.assertGreaterEqual(T, 5)

    # def test_circle_area(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #     ass4 = Assignment4()
    #     T = time.time()
    #     shape = ass4.fit_shape(sample=circ, maxtime=30)
    #     T = time.time() - T
    #     a = shape.area()
    #     self.assertLess(abs(a - np.pi), 0.01)
    #     self.assertLessEqual(T, 32)
    #
    # def test_bezier_fit(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #     ass4 = Assignment4()
    #     T = time.time()
    #     shape = ass4.fit_shape(sample=circ, maxtime=30)
    #     T = time.time() - T
    #     a = shape.area()
    #     self.assertLess(abs(a - np.pi), 0.01)
    #     self.assertLessEqual(T, 32)

    def test_circle_area_from_contour(self):
        circ = Circle(cx=1, cy=1, radius=1, noise=0.0)
        ass4 = Assignment4()
        T = time.time()
        a_computed = ass4.area(contour=circ.contour, maxerr=0.1)
        T = time.time() - T
        a_true = circ.area()
        # print(a_true)
        self.assertLess(abs((a_true - a_computed)/a_true), 0.1)


if __name__ == "__main__":
    unittest.main()
