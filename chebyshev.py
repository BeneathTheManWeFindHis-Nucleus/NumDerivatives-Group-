""" This is the main script for problem 2 of the Numerical Derivatives Group
homework. For my portion, I will attempt to approximate a numerical derivative
using Chebyshev methods in part a)."""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.special import roots_chebyt
from numpy.polynomial import Chebyshev as T

#Define functions
def f(x):                         #Function to be approximated
    return np.sin(x)

a = -np.pi                     #Determine domain
b = np.pi

def coefficients(N,k,j,f):      #Find coefficients
    coeffs = []
    for j_val in j:
        c_j = 0
        for k_val in k:
            eval_at = np.cos((np.pi*(k_val+.5))/N)
            c_j += f(eval_at)*(np.cos((np.pi*j_val*(k_val+.5))/N))
        c_j = c_j * (2 / N)
        if j_val == 0:
            c_j = c_j/2
        coeffs.append(c_j)
        print(f"c{j_val} = {c_j}") 
    return coeffs

#Initialize everything
n = 13                  #Order of the Chebyshev polynomial
N = n + 1               #Number of coefficients needed
k = np.arange(N)
j = np.arange(N)
print()

#Find Roots
roots, _ = roots_chebyt(n)
print(f"Roots of the Chebyshev polynomial of degree {n}:\n{roots}")
print()

#Find coefficients
print(f"The coefficients of the {n}-order polynomial are:")
coeffs = coefficients(N,k,j,f)
print()
# Define the interval
a = -np.pi
b = np.pi
x = (roots + 1) * (b - a) / 2 + a   # Map roots from [-1, 1] to [-π, π]

#Generate polynomial and evaluate it at the roots     
p = T(coeffs)
f_cheby = p(x) - (.5*coeffs[0])

#Plot the approximation
plt.figure(figsize=(10, 6))
plt.plot(x, f_cheby, marker='o', linestyle = '-', label='Chebyshev Approximation')
plt.plot(x, np.sin(x), linestyle='--',label='Sine Function')
plt.legend()
plt.xticks(np.arange(-np.pi, np.pi+0.1, np.pi/2), [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Chebyshev Interpolation for Sin(x)')
plt.grid(True)
plt.show()

#Calculate the derivative of the approximation and its error
p_prime = p.deriv()
f_cheby_prime = p_prime(x)
mean_deriv_error = abs(np.sum(np.cos(x) - f_cheby_prime)/n)
print(f"The mean error in the derivative of the {n}-order Chebyshev approximation is: {mean_deriv_error}.")

#Plot the derivative of the approximation vs cos(x)
plt.figure(figsize=(10, 6))
plt.plot(x, f_cheby_prime, marker='o', linestyle = '-', label='Chebyshev Derivative Approximation')
plt.plot(x, np.cos(x), linestyle='--',label='Cosine Function')
plt.legend()
plt.xticks(np.arange(-np.pi, np.pi+0.1, np.pi/2), [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Derivative of Chebyshev Interpolation for Sin(x)')
plt.grid(True)
plt.show() 
print()

def chebyshev(f,x):
    #start_time = time.time()
    n = 13
    N = n + 1
   
    k = np.arange(N)               #Find coeffiecients
    j = np.arange(N) 
    coeffs = []
    for j_val in j:
        c_j = 0
        for k_val in k:
            eval_at = np.cos((np.pi*(k_val+.5))/N)
            c_j += f(eval_at)*(np.cos((np.pi*j_val*(k_val+.5))/N))
        c_j = c_j * (2 / N)
        coeffs.append(c_j)
        
    roots, _ = roots_chebyt(n)        #Find roots, determine domain
    a = x[0]
    b = x[-1]
    x = (roots + 1) * (b - a) / 2 + a
    
    p = T(coeffs)                  #Determine Chebyshev polynomial and first derivative
    f_cheby = p(x)
    p_prime = p.deriv()
    f_cheby_prime = p_prime(x)
    
    mean_deriv_error = abs(np.sum(np.gradient(f(x),x) - f_cheby_prime)/n)   #Determine error
    #mean_deriv_error = abs(np.sum(np.cos(roots) - f_cheby_prime)/n) If true derivative is known, plug it in
    print(f"The mean error in the derivative of the {n}-order Chebyshev approximation is: {mean_deriv_error}.")

    #Plot the Chebyshev derivative vs the actual derivative
    plt.figure(figsize=(10, 6))
    plt.plot(x, f_cheby_prime, marker='o', linestyle = '-', label='Chebyshev Derivative Approximation')
    plt.plot(x, np.gradient(f(x),x), linestyle='--',label=f'Derivative of Arbitrary Function: f(x)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Derivative of Chebyshev Interpolation for Arbitrary Function')
    plt.grid(True)
    
    #end_time = time.time()
    #execution_time = end_time - start_time
    #print("Execution time:", execution_time, "seconds")

    plt.show() 
    
    return f_cheby_prime
   
cheb = chebyshev(f,x)

