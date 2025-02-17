#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 16 17:02 2025
Created in PyCharm
Created as sphenix_direct_photon/sinusoid_noise_fitting

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf


def main():
    # Simulated Data Parameters
    num_points = 6  # Number of data points to fit in each data set
    true_amplitude = 0.0  # Sinusoid signal amplitude. If at some point want to try actual signal
    frequency = 1.0  # Sinusoid signal frequency. If at some point want to try actual signal
    noise_std = 0.5  # Gaussian sigma for noise

    n_data = 1000  # Number of data sets to generate

    # Fit Parameters
    n_fits = 50  # Number of fits per generated data set
    fit_func = sinusoid_amp_phase_cos  # Fit function to use
    amplitude_range = np.linspace(-1.5, 1.5, 50)  # Range of amplitudes for randomly setting p0 of fit
    p0 = [1.0, 0.0]  # Initial fit parameters (to be replaced)


    amps, amps_0, amp_stds, phases, sin_amps, cos_amps = [], [], [], [], [], []
    for j in range(n_data):
        print(f'Data Set: {j + 1}')
        # Generate data
        x, y_noisy = generate_data(num_points, true_amplitude, frequency, noise_std)

        # Fit data multiple times for random initial parameters
        fit_sin_amps, fit_cos_amps, fit_amps, fit_phases, chi2_per_dofs = [], [], [], [], []
        for i in range(n_fits):
            rand_amp_sin = np.random.uniform(min(amplitude_range), max(amplitude_range))
            rand_amp_cos = np.random.uniform(min(amplitude_range), max(amplitude_range))
            p0[0], p0[1] = rand_amp_sin, rand_amp_cos
            popt, perr = fit_data(x, y_noisy, p0, func=fit_func, bounds=None)
            chi2_per_dof_sin = chi_square(y_noisy, fit_func(x, *popt), noise_std) / (num_points - 2)
            phase = np.arctan(popt[1] / popt[0])
            # amp = popt[0] / np.cos(phase)
            amp = np.sqrt(popt[0]**2 + popt[1]**2)
            fit_amps.append(amp)
            fit_phases.append(phase)
            fit_sin_amps.append(popt[0])
            fit_cos_amps.append(popt[1])
            chi2_per_dofs.append(chi2_per_dof_sin)

        # Take the fit with the lowest chi-square
        min_chi2_index = np.argmin(chi2_per_dofs)
        min_amp, min_phase = fit_amps[min_chi2_index], fit_phases[min_chi2_index]
        min_sin_amp, min_cos_amp = fit_sin_amps[min_chi2_index], fit_cos_amps[min_chi2_index]
        chi2_per_dof_sin = chi_square(y_noisy, fit_func(x, min_sin_amp, min_cos_amp), noise_std) / (num_points - 2)

        # Compare to chi2 for flat line
        chi2_per_dof_0 = chi_square(y_noisy, np.zeros_like(y_noisy), noise_std) / num_points
        print(f'Chi-Square per DOF Sinusoid: {chi2_per_dof_sin:.2f}, Chi-Square per DOF 0: {chi2_per_dof_0:.2f}')

        if np.std(fit_amps) > 0:
            print(f'Local minima in landscape, Std Dev Amplitude: {np.std(fit_amps):.2f}')
        amps.append(min_amp)
        amp_stds.append(np.std(fit_amps))
        phases.append(min_phase)
        sin_amps.append(min_sin_amp)
        cos_amps.append(min_cos_amp)
        if chi2_per_dof_0 < chi2_per_dof_sin:
            amps_0.append(0)
        else:
            amps_0.append(min_amp)

    # Plot amp distribution
    plt.figure(figsize=(8, 6))
    plt.hist(amps, bins=50)
    plt.xlabel("Amplitude")
    plt.ylabel("Counts")
    plt.title("Amplitude Distribution")

    plt.figure(figsize=(8, 6))
    plt.hist(amps_0, bins=50)
    plt.xlabel("Amplitude")
    plt.ylabel("Counts")
    plt.title("Amplitude Distribution for Flat Line or Sinusoid")

    plt.figure(figsize=(8, 6))
    plt.hist(phases, bins=50)
    plt.xlabel("Phases")
    plt.ylabel("Counts")
    plt.title("Phase Distribution")

    plt.figure(figsize=(8, 6))
    plt.hist(sin_amps, bins=50)
    plt.xlabel("Sin Amplitudes")
    plt.ylabel("Counts")
    plt.title("Distribution of Sin Amplitudes")

    plt.figure(figsize=(8, 6))
    plt.hist(cos_amps, bins=50)
    plt.xlabel("Cos Amplitudes")
    plt.ylabel("Counts")
    plt.title("Distribution of Cos Amplitudes")

    # Plot 2D scatter of amplitude vs std dev of amplitude
    plt.figure(figsize=(8, 6))
    plt.scatter(amps, amp_stds, alpha=0.5)
    plt.xlabel("Amplitude")
    plt.ylabel("Std Dev Amplitude")
    plt.title("Amplitude vs Std Dev Amplitude")

    plt.show()
    print('donzo')


def generate_data(num_points=100, amplitude=1.0, frequency=1.0, noise_std=0.2):
    """Generates noisy sinusoidal data."""
    x = np.linspace(0, 2 * np.pi, num_points)
    y_true = amplitude * np.sin(frequency * x)
    noise = np.random.normal(0, noise_std, size=num_points)
    y_noisy = y_true + noise
    return x, y_noisy


def plot_data(x, y_data, y_model=None):
    """Plots data and model."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y_data, label='Data', color='b')
    if y_model is not None:
        plt.plot(x, y_model, label='Model', color='r')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data and Model")
    plt.legend()
    plt.grid()


def fit_data(x, y_data, p0, func=None, bounds=''):
    """Fits data to a sinusoidal model."""
    if func is None:
        func = sinusoid_amp if len(p0) == 1 else sinusoid_amp_phase
    if bounds == '':
        bounds = ([-np.inf], [np.inf]) if len(p0) == 1 else ([-np.inf, 0], [np.inf, np.pi / 2])
    elif bounds is None:
        bounds = (-np.inf, np.inf)

    popt, pcov = cf(func, x, y_data, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def chi_square(y_data, y_model, error):
    """Computes the chi-square statistic."""
    return np.sum(((y_data - y_model) / error) ** 2)


def compute_chi_square(x, y_data, amplitude_range, phase_shifts, noise_std=0.2):
    """Computes chi-square for different amplitudes and phase shifts."""
    chi2_results = {}
    for phase in phase_shifts:
        chi2_values = []
        for amplitude in amplitude_range:
            y_model = sinusoid_amp_phase(x, amplitude, phase)
            chi2_values.append(chi_square(y_data, y_model, noise_std))
        chi2_results[phase] = chi2_values
    return chi2_results


def sinusoid_amp_phase(x, amplitude, phase):
    """Sinusoidal model."""
    return amplitude * np.sin(x + phase)


def sinusoid_amp_phase_cos(x, amp_sin, amp_cos):
    return amp_sin * np.sin(x) + amp_cos * np.cos(x)


def sinusoid_amp(x, amplitude):
    """Sinusoidal model."""
    return amplitude * np.sin(x)


if __name__ == '__main__':
    main()

