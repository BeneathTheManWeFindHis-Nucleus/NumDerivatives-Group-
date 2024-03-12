""" This is the main script for problem 2 of the Numerical Derivatives Group
homework. For my portion, I will attempt to approximate a numerical derivative
using Chebyshev methods in part a)."""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import roots_chebyt
from numpy.polynomial import Chebyshev as T

#Define functions
def y(x):               #Function to be approximated
    return np.sin(x)

def coefficients(N,k,j,y):      #Find coefficients
    coeffs = []
    for j_val in j:
        c_j = 0
        for k_val in k:
            eval_at = np.cos((np.pi*(k_val+.5))/N)
            c_j += y(eval_at)*(np.cos((np.pi*j_val*(k_val+.5))/N))
        c_j = c_j * (2 / N)
        coeffs.append(c_j)
        print(f"c{j_val} = {c_j}") 
    return coeffs

#Initialize everything
n = 11                  #Order of the Chebyshev polynomial
N = n + 1              #Number of coefficients needed
k = np.arange(N)
j = np.arange(N)
print()

#Find Roots
roots, _ = roots_chebyt(n)
print(f"Roots of the Chebyshev polynomial of degree {n}:\n{roots}")
print()

#Find coefficients
print(f"The coefficients of the {n}-order polynomial are:")
coeffs = coefficients(N,k,j,y)
print()
# Define the interval
a = -np.pi
b = np.pi
x = (roots + 1) * (b - a) / 2 + a   # Map roots from [-1, 1] to [-π, π]

#Generate polynomial and evaluate it at the roots     
p = T(coeffs)
y_cheby = p(roots) 

#Plot the approximation
plt.figure(figsize=(10, 6))
plt.plot(x, y_cheby, marker='o', linestyle = '-', label='Chebyshev Approximation')
plt.plot(x, np.sin(roots), linestyle='--',label='Sine Function')
plt.legend()
plt.xticks(np.arange(-np.pi, np.pi+0.1, np.pi/2), [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Chebyshev Interpolation for Sin(x)')
plt.grid(True)
plt.show()

#Calculate the derivative of the approximation and its error
p_prime = p.deriv()
y_cheby_prime = p_prime(roots)
mean_deriv_error = np.sum(np.cos(roots) - y_cheby_prime)/(len(roots))
print(f"The mean error in the derivative of the {n}-order Chebyshev approximation is: {mean_deriv_error}.")

#Plot the derivative of the approximation vs cos(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y_cheby_prime, marker='o', linestyle = '-', label='Chebyshev Derivative Approximation')
plt.plot(x, np.cos(roots), linestyle='--',label='Cosine Function')
plt.legend()
plt.xticks(np.arange(-np.pi, np.pi+0.1, np.pi/2), [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Derivative of Chebyshev Interpolation for Sin(x)')
plt.grid(True)
plt.show() 
