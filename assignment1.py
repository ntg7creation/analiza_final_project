"""
In this assignment you should interpolate the given function.
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


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    # Python3 program for implementation
    # of Lagrange's Interpolation


    #Lagrange Polynomials
    def getinnerterm(self, j: int, i: int, pointlist: list) -> callable:
        if j == -1:
            return lambda x: 1;
        term = self.getinnerterm(j - 1, i, pointlist);
        if j == i:
            return term;
        if(pointlist[i].x - pointlist[j].x) == 0: #some wird error i thikn the values are just too small - without this it takes 2 min to run the basic tests
            return term;
        return lambda _x: term(_x) * (_x - pointlist[j].x) / (pointlist[i].x - pointlist[j].x);

        pass

    def getresult(self,n:int,i:int,pointlist:list)->callable:
        if i== -1:
            return lambda x:0;
        preresult = self.getresult(n,i-1,pointlist);
        term = self.getinnerterm(n-1, i, pointlist);
        return  lambda x: preresult(x) + pointlist[i].y * term(x);
        pass

    def interpolate2(self, f: callable, a: float, b: float, n: int) -> callable:
        pointlist = [Point] * n;
        for i in range(0, n):
            pointlist[i].x = a + (abs(a - b) / n) * i
            pointlist[i].y = f(pointlist[i].x)
        result = self.getresult(n,n-1,pointlist);
        # result = lambda x: 0;
        # for i in range(n):
        #     # term = 1;
        #     # for j in range(n):
        #     #     if j != i:
        #     #         term = lambda _x: term * (_x - pointlist[j].x) / (pointlist[i].x - pointlist[j].x);
        #     term = self.getinnerterm(n, i, pointlist);
        #
        #     result = lambda x: result(x) + pointlist[i].y * term(x);

        return result

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:


        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """


        # replace this line with your solution to pass the second test
        result = self.interpolate2(f, a, b, n);
        #result = lambda x:x;

        return result


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
