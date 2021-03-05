#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:04:36 2020

@author: Arlo Parker
"""
import numpy as np
from numpy import linalg as LA

def CGM(A,x,b,target):       
    r = b - A.dot(x)     #3x1 
    #print(r)
    r_magnitude = (np.linalg.norm(r))      #scalar
    print(r_magnitude)
    
    if (r_magnitude<target):    
        return (x)
    else:
        r_T = np.array(r).T     #1x3          
        next_search_dir = r             #3x1        
        next_search_dir_T = np.array(r).T      #1x3 
        
        alpha = np.dot(r_T,r)/np.dot(next_search_dir_T,A.dot(next_search_dir))     #1x1 [[alpha]]
        alpha = np.asscalar(np.array([alpha]))       #scalar
        #print(alpha)
        
        x = x + alpha * next_search_dir    #3x1 
        #print(x)
        
#    ************************************** THIS COMPLETES 1st ITERATION *****************************************  
        r_next = r - alpha * A.dot(next_search_dir)  #3x1 
        #print(r_next)
        r_next_T = np.array(r_next).T       #1x3       
        r_next_magnitude = (np.linalg.norm(r_next))       #scalar
        #print(r_next_magnitude)
   
#    # ************************************* THE ALGORITHM STARTS HERE *******************************************
        count = 1
        while (r_next_magnitude > target):  
            count += 1
            
            beta = np.dot(r_next_T,r_next)/np.dot(r_T,r)       #1x1 [[alpha]]                      
            beta = np.asscalar(np.array([beta]))    #scalar
            #print(beta)
            
            next_search_dir = r_next + beta * next_search_dir   #3x1 
            next_search_dir_T = np.array(next_search_dir).T          #1x3
            #print(next_search_dir)
            
            alpha = np.dot(r_next_T,r_next)/np.dot(next_search_dir_T,A.dot(next_search_dir))      #1x1 
            alpha = np.asscalar(np.array([alpha]))        #scalar
            #print(alpha)
            
            x = x + alpha * next_search_dir     #3x1 
            #print(x)
            #print(r_next)
            r_next = r_next - alpha * A.dot(next_search_dir)  #3x1 
            #print(r_next)
            r_next_T = np.array(r_next).T       #1x3       
            r_next_magnitude = (np.linalg.norm(r_next))       #scalar
           
            #print(r_next_magnitude)
        
        print(x)
        print(count)
            
        
        
        
"""
Bx = c =====> (B^T*B)x = B^T*c =====> Ax = b
"""
B = np.asarray([[3,2,-1], 
                [2,-2,4],
                [-1,.5,-1]])   

B_transpose = B.T
A = np.array(B_transpose.dot(B))    #A = symmetric 

w, v = LA.eig(B)    #check eigenvalues to see if A is positive definite  
for i in w:
    if (i > 0):
        continue
    else:
        print("The matrix is not positive definite")

c = np.asarray([[1,-2,0]]).T      #3x1  
b = np.array(B_transpose.dot(c))         #3x1 
print(A)
print(b)

x = np.asarray([[0,0,0]]).T   #initial guess 3x1 
    
CGM(A,x,b,.0000001)






