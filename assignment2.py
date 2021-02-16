"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Point(object):
    def __init__(self):
        """ Create a new point at the origin """
        self.x: float = 0.0
        self.y: float = 0.0

    pass

class dualPoint(object):
    x_2:float = 0.0
    x_1:float = 0.0

    def __init__(self):
        """ Create a new point at the origin """

    pass

class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass





    def the_secant_method(self,xi_2:float,xi_1:float,f:callable,maxerr:float)->float:
        #using xi_2 as x[i-2]
        xi = xi_1 - f(xi_1)*(xi_1-xi_2)/(f(xi_1)-f(xi_2))
        #dont know sytacks for do while
        while abs(f(xi)) >= maxerr :
            xi_2=xi_1
            xi_1=xi
            xi = xi_1 - f(xi_1) * (xi_1 - xi_2) / (f(xi_1) - f(xi_2))

        return xi
        pass

    def intersections2(self, f1: callable, f2: callable, a: float, b: float,n:int, maxerr ) -> list:

        f=lambda x:f1(x)-f2(x)


        result =[]
        counter =0
        pointlist = []



        prePoint = a + (abs(a - b) / n) * 0
        positive_neg_detector = np.sign(f(prePoint))

        for i in range(1,n):
            postPoint = a + (abs(a - b) / n) * i

            if np.sign(f(postPoint)) != positive_neg_detector:
                positive_neg_detector *=-1

                pointlist.append( [prePoint,postPoint])
                counter += 1
            prePoint = postPoint

        for i in range(0,counter):
            result.append(self.the_secant_method(pointlist[i][0],pointlist[i][1],f,maxerr))
        print (result)
        return  result
        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:

        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution
        X=self.intersections2(f1, f2, a, b, 100,maxerr)
        return X


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
