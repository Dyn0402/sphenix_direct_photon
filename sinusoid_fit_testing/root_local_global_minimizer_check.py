#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 13 15:22 2025
Created in PyCharm
Created as Misc/root_local_global_minimzer_check

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf


def main():
    # Generate x values
    x = np.linspace(-3, 3, 400)

    # Evaluate polynomial
    y = poly6(x)

    poly6_wrapper = lambda a, x: poly6(x)

    x_data = [0]
    data = [-10]
    p0 = [-2.0]

    popt, pcov = cf(poly6_wrapper, x_data, data, p0=p0)
    print(popt)

    # Plot the polynomial
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='6th order polynomial', color='b')
    plt.title('5th Order Polynomial with Global and Local Minimum')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.axvline(p0[0], color='g', label='Initial Guess')
    plt.axvline(popt[0], color='r', label='Minimum')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.show()
    print('donzo')


# Define a 5th order polynomial with both a global and local minimum
def poly6(x):
    return x**6 - 0.14 * x**5 - 5*x**4 + 6*x**3 + 2*x**2 - 3*x + 1


if __name__ == '__main__':
    main()
