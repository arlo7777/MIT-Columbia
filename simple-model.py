#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:14:46 2020

@author: Arlo Parker
"""
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt

class Matrix:
    
#    ************************************************* STEP 1 **************************************************
#    construct the matrices using Finite Difference Method

    def __init__(self, p, D_t, x_1, r, δ, π, σ):
        self.p = p
        self.D_t = D_t
        self.x_1 = x_1 
        self.r = r
        self.delta = δ
        self.pi = π
        self.sigma = σ
        
    def x_column_vector_constructor(self):
        
        x = [self.x_1]
        
        for i in range(2,self.p+1):
            x_i = self.x_1 + (i - 1)*self.D_t
            x.append(round(x_i,10))
        
        return np.array(x).T

    def d_forward_constructor(self):
        
        D_forward = np.eye(self.p,self.p)*round(-1/self.D_t, 10) + np.eye(self.p,self.p,k=1)*round(1/self.D_t, 1)
        D_forward[self.p-1,self.p-1] = round(1/self.D_t, 10) #change the right bottom value to 1/D_t 
        D_forward[self.p-1, self.p-2] = round(-1/self.D_t, 10)   #add -1/D_t rounded to 1 decimal at this position 
        
        return D_forward
    
    def d_backward_constructor(self):
        
        D_backward = np.eye(self.p,self.p,k=-1)*round(-1/self.D_t, 10) + np.eye(self.p,self.p)*round(1/self.D_t, 10)
        D_backward[0,0] = round(-1/self.D_t, 10) #change the value in this position to -1/D_t
        D_backward[0,1] = round(1/self.D_t, 10)  #add 1/D_t at this position 
        
        return D_backward
        
    def d_second_differencing_constructor(self):
           
        D_second_differencing = np.eye(self.p,self.p,k=-1)*round(1/(self.D_t**2), 10)+ np.eye(self.p,self.p)*round(-2/self.D_t**2, 10) + \
        np.eye(self.p,self.p,k=1)*round(1/(self.D_t**2), 10)
        
        D_second_differencing[0,1] = round(-2/(self.D_t**2), 10)   #change the value at this position to -2/D_t
        D_second_differencing[self.p-1,self.p-2] = round(-2/(self.D_t**2), 10)   
        D_second_differencing[0,2] = round(1/self.D_t**2, 10)   #add an aditional value at this position
        D_second_differencing[self.p-1,self.p-3] = round(1/self.D_t**2, 10)  
        D_second_differencing[0,0] = round(1/self.D_t**2, 10)    #change the value at this position to 1/D_t
        D_second_differencing[self.p-1,self.p-1] = round(1/self.D_t**2, 10)
        
        return D_second_differencing


#    ********************************************** STEP 2 *************************************************
#    construct matrix B and vector c to use as input for the Conjugate Gradient Method     

    def B_constructor(self, D, D_second_differencing):
        
        B_1 = (1/self.D_t + self.r) * np.identity(100)     #matrix pxp
        
        k_diag_values = []                
        for i in range(0, 100):
            k_diag_values.append(i)               
        k_diag = np.diag(k_diag_values)                    
        B_2 = np.array(self.delta * k_diag.dot(D))      #matrix pxp
        
        k_diag_squared = k_diag * k_diag
        sigma_times_k = round(self.sigma**2/2, 1) * k_diag_squared  
        B_3 = np.array(sigma_times_k.dot(D_second_differencing))       #matrix pxp

        B = B_1 + B_2 + B_3
        return B      #matrix pxp
    
    def v_init(self):
        v_init = np.zeros((100, 1))       ##vertical vector px1
        return v_init   
    
    def k_constructor(self):
        
        k_vector = [] 
        
        for i in range(0, 100):
            k_vector.append(i)               

        k = np.array([k_vector]).T
        return k
        
    def c_constructor(self, v_init):
        
        c = 1/self.D_t * v_init + self.pi * simple_model.k_constructor()     #vertical vector px1
        return c
    
    def c_update(self, v_new):
        
        c_new = 1/self.D_t * v_new + self.pi * simple_model.k_constructor()       #vertical vector px1
        return c_new


#    ************************************************* STEP 3 *************************************************
#    apply CG algorithm to find v_new at convergence
        
    def CGM(self,A,v_old,b_old,target):  
        
        count = 0
        
        while (count <= 2):
        
            r = b_old - A.dot(v_old)     #px1 
            r_magnitude = (np.linalg.norm(r))      #scalar
        
            if (r_magnitude<target):    
                return (v_old)
            else:
                r_T = np.array(r).T     #1xp          
                next_search_dir = r             #px1        
                next_search_dir_T = np.array(r).T      #1xp
                
                alpha = np.dot(r_T,r)/np.dot(next_search_dir_T,A.dot(next_search_dir))     #1x1 [[alpha]]
                alpha = np.asscalar(np.array([alpha]))       #scalar
                
                v_new = v_old + alpha * next_search_dir    #px1 
                b_new = simple_model.c_update(v_new)       #update b with v_new 
                
                b_new = np.array(B_transpose.dot(b_new))      #multiply both sides by B'(right side)
                A = np.array(B_transpose.dot(A))          #multiply both sides by B'(left side)
            
                b_old = b_new
                v_old = v_new
                #print(v_new)      #v is a vector of size k 
                count += 1
                
                               
        #print(np.round(v_new, 5))
        print('\n', count)       
        
       
        return v_new
        
    
    
#    *************************************************** STEP 4 *************************************************
#    plot v_new as a function of k
        
    def plot_function(self, k, v_new):
        
        combined = np.vstack((np.array([k]),np.array([v_new]))).T
        
        #print(combined)
        
        x, y = combined.T
        plt.scatter(x,y)
        plt.show()       #plot the pairs
        
 
    
    
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    

    
    
#    *************************************************** STEP 1 **************************************************
#    construct the matrices using Finite Difference Method
        
simple_model = Matrix(100, float(1), 0, .02, .01, .06, .1)

x_column_vector = simple_model.x_column_vector_constructor()

D_forward = simple_model.d_forward_constructor()
D_backward = simple_model.d_backward_constructor()
D_second_differencing = simple_model.d_second_differencing_constructor()



#    *************************************************** STEP 2 ***************************************************
#    construct matrix B and vector c to use as input for the Conjugate Gradient Method 


'''
B = [(1/D_t + r) * I + δ * diag(k) * D_matrix - (σ^2/2) * diag(k^2) * D_second_derivative]
c = 1/D_t * v_old + π * k
'''
B = simple_model.B_constructor(D_forward, D_second_differencing)      #matrix pxp
c = simple_model.c_constructor(simple_model.v_init())       #vertical vector px1
            


#    *************************************************** STEP 3 **************************************************
#    use CG algorithm to find v_new at convergence

"""
Bx = c =====> (B^T*B)x = B^T*c =====> Ax = b
"""

B_transpose = B.T
A = np.array(B_transpose.dot(B))    #A = symmetric 

w, v = LA.eig(B)    #check eigenvalues to see if A is positive definite  
for i in w:
    if (i > 0):
        continue
    else:
        print("The matrix is not positive definite")

b = np.array(B_transpose.dot(c))         #vertical vector px1 

v_old = simple_model.v_init()   #initial guess vertical vector px1 

v_new = simple_model.CGM(A,v_old,b,.0000000001)



#    *************************************************** STEP 4 ****************************************************
#    plot v_new as a function of k

simple_model.plot_function(v_new, simple_model.k_constructor())



#    *************************************************** STEP 5 ****************************************************
#repeat with D_backward -- the plot should not change

B = simple_model.B_constructor(D_backward, D_second_differencing)      #matrix pxp
c = simple_model.c_constructor(simple_model.v_init())             #vertical vector px1


#repeat step 3 and 4


B_transpose = B.T
A = np.array(B_transpose.dot(B))    #A = symmetric 

w, v = LA.eig(B)    #check eigenvalues to see if A is positive definite  
for i in w:
    if (i > 0):
        continue
    else:
        print("The matrix is not positive definite")

b = np.array(B_transpose.dot(c))         #vertical vector px1 

v_old = simple_model.v_init()   #initial guess vertical vector px1 

v_new = simple_model.CGM(A,v_old,b,.0000001)

simple_model.plot_function(v_new, simple_model.k_constructor())

