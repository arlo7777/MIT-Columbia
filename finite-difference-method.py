#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:30:11 2020

@author: Arlo Parker
"""
import numpy as np
from matplotlib import pyplot as plt

class Matrix:

    def __init__(self, p, delta, x_1):
        self.p = p
        self.delta = delta
        self.x_1 = x_1         
        
    
    def d_forward_constructor(self, p, delta):
        
        D_forward = np.eye(p,p)*round(-1/delta, 10) + np.eye(p,p,k=1)*round(1/delta, 1)
        D_forward[p-1,p-1] = round(1/delta, 10) #change the right bottom value to 1/delta 
        D_forward[p-1, p-2] = round(-1/delta, 10)   #add -1/delta rounded to 1 decimal at this position 
        
        return D_forward
    
    
    def d_backward_constructor(self, p, delta):
        
        D_backward = np.eye(p,p,k=-1)*round(-1/delta, 10) + np.eye(p,p)*round(1/delta, 10)
        D_backward[0,0] = round(-1/delta, 10) #change the value in this position to -1/delta
        D_backward[0,1] = round(1/delta, 10)  #add 1/delta at this position 
        
        return D_backward
    
    
    def d_second_differencing_constructor(self, p, delta):
           
        D_second_differencing = np.eye(p,p,k=-1)*round(1/(delta**2), 10)+ np.eye(p,p)*round(-2/delta**2, 10) + \
        np.eye(p,p,k=1)*round(1/(delta**2), 10)
        
        D_second_differencing[0,1] = round(-2/(delta**2), 10)   #change the value at this position to -2/delta
        D_second_differencing[p-1,p-2] = round(-2/(delta**2), 10)   
        D_second_differencing[0,2] = round(1/delta**2, 10)   #add an aditional value at this position
        D_second_differencing[p-1,p-3] = round(1/delta**2, 10)  
        D_second_differencing[0,0] = round(1/delta**2, 10)    #change the value at this position to 1/delta
        D_second_differencing[p-1,p-1] = round(1/delta**2, 10)
        
        return D_second_differencing
    
    
    def x_column_vector_constructor(self, p, delta, x_1):
        
        x = [x_1]
        
        for i in range(2,p+1):
            x_i = x_1 + (i - 1)*delta
            x.append(round(x_i,10))
        
        return np.array(x).T


    def f_column_vector_constructor(self, p, delta, x_1):
        
        x = [x_1]   #x_1 is given, it's the first value in the list
        f = []
        
        for i in range(2,p+1):
            x_i = x_1 + (i - 1)*delta    #formula for the column vector x
            x.append(round(x_i,10))   #append x_i values to the list x 
        
        for i in x:     #iterate through values in list x
            y = (1/2)*(i**2)    #plug values x into the mathematical function
            f.append(round(y,10))     #append the result to the list f
        
        return np.array(f).T
    
       
    def matrix_multiplication_forward(self, X, D, F):
        
        combined = np.vstack((np.array([X]), np.array([np.array(D.dot(F)).T]))).T     #combine fp, tp; make (fp, tp) pairs
        
        print(combined)
        
        x, y = combined.T    #allocate the values for the plot 
        plt.scatter(x,y)    #construct a plot
        plt.show()     #show the plot
        print("\n\n")
    
        
    def matrix_multiplication_backward(self, X, D, F):
        
        combined = np.vstack((np.array([X]), np.array([np.array(D.dot(F)).T]))).T
        
        print(combined)
        
        x, y = combined.T
        plt.scatter(x,y)
        plt.show()         #display the plot
        print("\n\n")

    
    def matrix_multiplication_second_differencing(self, X, D, F):
        
        combined = np.vstack((np.array([X]), np.array([np.array(D.dot(F)).T]))).T
        
        print(combined)
        
        x, y = combined.T
        plt.scatter(x,y)
        plt.show()       #plot the pairs
    
   
difference_approximation = Matrix(101, float(.001), 0) 

x_column_vector = difference_approximation.x_column_vector_constructor(101, float(.001), 0)
f_column_vector = difference_approximation.f_column_vector_constructor(101, float(.001), 0)
D_forward = difference_approximation.d_forward_constructor(101, float(.001))
D_backward = difference_approximation.d_backward_constructor(101, float(.001))
D_second_differencing = difference_approximation.d_second_differencing_constructor(101, float(.001))

difference_approximation.matrix_multiplication_forward(x_column_vector, D_forward, f_column_vector)
difference_approximation.matrix_multiplication_backward(x_column_vector, D_backward, f_column_vector)
difference_approximation.matrix_multiplication_second_differencing(x_column_vector, D_second_differencing, f_column_vector)
