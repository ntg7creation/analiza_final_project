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

class Point(object):
    def __init__(self):
        """ Create a new point at the origin """
        self.x: float = 0.0
        self.y: float = 0.0

    pass

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
        n = 55
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
        # print("area1:")
        # print(resulta)

        n = 77
        matrixa = contour(n)
        half =(int)(n/2)
        firstpart= matrixa[half:]
        secondpart= matrixa[:half+1]
        fliped = np.copy(secondpart)
        for i in range(0,half+1):
            fliped[i] = (secondpart[half - i])
        resultb = self.areabetween2(fliped,firstpart,half+1)
        # print("area2:")
        # print(resultb)

        dn = 77 - 55
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
        # print("area3:")
        # print(resultc)



        # matrixa = np.transpose(matrixa)
        #
        # xarray = matrixa[0]
        # yarray = matrixa[1]

        # print(xarray)
        # print(yarray)
        # print (contour(13))
        # array = []
        # for i in range (13):
        #     array.append(contour(i))
        # print(array)

        return np.float32(resulta)


    def Quicksort2D_y(self, array: list, number: int, center: float,needflip:bool):

        if number < 2 :
            return  array
        upcounter =0
        upavg = 0.0
        downcounter = 0
        downavg = 0.0
        array_up = []
        array_down = []
        for i in range(number) :
            if array[i].y >= center:
                upcounter +=1
                upavg += (array[i].x - upavg)/upcounter
                array_up.append(array[i])
            else:
                downcounter += 1
                downavg += (array[i].x - downavg) / downcounter
                array_down.append(array[i])

        sortedup = self.Quicksort2D_x(array_up,upcounter,upavg,False)
        sorteddown = self.Quicksort2D_x(array_down,downcounter,downavg,False)
        fdown = sorteddown[0]
        ldown = sorteddown[-1]
        if(abs(fdown.y-number)<abs(ldown.y-number)):
            flip =[]
            for i in range(downcounter):
                flip.append(sorteddown[downcounter-1-i])
            sorteddown = flip
        fup = sortedup[0]
        lup = sortedup[-1]
        if(abs(fup.y-number)<abs(lup.y-number)):
            flip =[]
            for i in range(upcounter):
                flip.append(sortedup[upcounter-1-i])
            sortedup = flip
        finalarray =   sorteddown +sortedup

        # finalarray.append(sortedup)
        # finalarray.append(sorteddown)
        return finalarray
        pass


    def Quicksort2D_x(self,array:list,number:int,center:float,needflip:bool):

        if number < 2 :
            return  array
        temp:Point = array[1]
        leftcounter =0
        leftavg = 0.0
        rightcounter = 0
        rightavg = 0.0
        array_left = []
        array_right = []
        for i in range(number) :
            if array[i].x <= center:
                leftcounter +=1
                leftavg += (array[i].y - leftavg)/leftcounter
                array_left.append(array[i])
            else:
                rightcounter += 1
                rightavg += (array[i].y - rightavg) / rightcounter
                array_right.append(array[i])

        sortedleft = self.Quicksort2D_y(array_left,leftcounter,leftavg,False)
        sortedright = self.Quicksort2D_y(array_right,rightcounter,rightavg,False)
        fright=sortedright[0]
        lright=sortedright[-1]
        if(abs(lright.x-number)<abs(fright.x-number)):
            flip =[]
            for i in range(rightcounter):
                flip.append(sortedright[rightcounter-1-i])
            sortedright = flip
        fleft = sortedleft[0]
        lleft = sortedleft[-1]
        if(abs(lleft.x-number)>abs(fleft.x-number)):
            flip =[]
            for i in range(leftcounter):
                flip.append(sortedleft[leftcounter-1-i])
            sortedleft = flip
        finalarray =sortedleft + sortedright

        # finalarray.append(sortedleft)
        # finalarray.append(sortedright)

        return finalarray
        pass

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
        start =time.time()
        counter =0
        sumx = 0.0
        averagex = 0.0

        Pointlist = []
        Pointlistx = []
        Pointlisty = []
        # while True:
        for i in range (64):
            temp =  sample()
            x, y = temp
            Pointlist.append(Point())
            Pointlist[i].x = x
            Pointlist[i].y = y
            counter += 1
            averagex += (x - averagex)/counter
            now = time.time()
            if now - start   > maxtime/4 :
                break
        sortedarray = self.Quicksort2D_x(Pointlist,counter,averagex,True)
        for i in range (64):
             Pointlistx.append(sortedarray[i].x)
             Pointlisty.append(sortedarray[i].y)
        # print(Pointlistx)
        # print(Pointlisty)
        # plt.plot(Pointlistx, Pointlisty, '-', color='black');
        # plt.show()

        result = MyShape()
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

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)
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

    # def test_circle_area_from_contour(self):
    #     circ = Circle(cx=1, cy=1, radius=1, noise=0.0)
    #     ass4 = Assignment4()
    #     T = time.time()
    #     a_computed = ass4.area(contour=circ.contour, maxerr=0.1)
    #     T = time.time() - T
    #     a_true = circ.area()
    #     # print(a_true)
    #     self.assertLess(abs((a_true - a_computed)/a_true), 0.1)


if __name__ == "__main__":
    unittest.main()
