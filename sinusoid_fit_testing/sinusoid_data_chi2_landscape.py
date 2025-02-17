#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 13 17:25 2025
Created in PyCharm
Created as Misc/sinusoid_data_chi2_landscape

@author: Dylan Neff, dn277127
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cf


def main():
    # original_func()
    sin_cos_func()

    plt.show()
    print('donzo')

def original_func():
    # Parameters
    num_points = 6
    true_amplitude = 0.0
    frequency = 1.0 * 10000
    noise_std = 0.5
    amplitude_range = np.linspace(-1.5, 1.5, 50)
    phase_shifts = np.linspace(-2, +2, 5)  # Testing different phase shifts
    n_fits = 50
    n_data = 1000
    fit_func = sinusoid_amp_phase
    p0 = [1.0, 0.0]

    amps, amps_0, amp_stds = [], [], []
    for j in range(n_data):
        print(f'Data Set: {j + 1}')
        # Generate data
        x, y_noisy = generate_data(num_points, true_amplitude, frequency, noise_std)

        # Fit data multiple times for random initial parameters
        # p0 = [1.0]
        fit_amps, fit_phases, chi2_per_dofs = [], [], []
        for i in range(n_fits):
            rand_amp = np.random.uniform(min(amplitude_range), max(amplitude_range))
            p0[0] = rand_amp
            popt, perr = fit_data(x, y_noisy, p0)
            chi2_per_dof_sin = chi_square(y_noisy, fit_func(x, *popt), noise_std) / (num_points - 2)
            # chi2_per_dof_sin = chi_square(y_noisy, sinusoid_amp(x, *popt), noise_std) / (num_points - 1)
            # print(f'Amplitude: {popt[0]:.2f} +/- {perr[0]:.2f}')
            # print(f'Phase: {popt[1]:.2f} +/- {perr[1]:.2f}')
            fit_amps.append(popt[0])
            # fit_phases.append(popt[1])
            fit_phases.append(0)
            chi2_per_dofs.append(chi2_per_dof_sin)
        # Take the fit with the lowest chi-square
        min_chi2_index = np.argmin(chi2_per_dofs)
        popt = [fit_amps[min_chi2_index], fit_phases[min_chi2_index]]
        chi2_per_dof_sin = chi_square(y_noisy, fit_func(x, *popt), noise_std) / (num_points - 2)
        chi2_per_dof_0 = chi_square(y_noisy, np.zeros_like(y_noisy), noise_std) / num_points
        print(f'Chi-Square per DOF Sinusoid: {chi2_per_dof_sin:.2f}, Chi-Square per DOF 0: {chi2_per_dof_0:.2f}')

        # print(f'Std Dev Amplitude: {np.std(fit_amps):.2f}')
        # print(f'Std Dev Phase: {np.std(fit_phases):.2f}')
        if np.std(fit_amps) > 0:
            print(f'Std Dev Amplitude: {np.std(fit_amps):.2f}')
        amps.append(popt[0])
        amp_stds.append(np.std(fit_amps))
        if chi2_per_dof_0 < chi2_per_dof_sin:
            amps_0.append(0)
        else:
            amps_0.append(popt[0])

        # if np.std(fit_amps) > 0.2:
        if np.std(fit_amps) > 200.2:
            # Plot 2D histogram of fit results
            plt.figure(figsize=(8, 6))
            amp_bin_edges = np.linspace(-1.0, 1.0, 50)
            phase_bin_edges = np.linspace(-np.pi, np.pi, 50)
            hist, xedges, yedges = np.histogram2d(fit_amps, fit_phases, bins=[amp_bin_edges, phase_bin_edges])

            # Create a colormap with white for the lowest values
            cmap = plt.cm.jet
            cmap.set_under('white')

            plt.imshow(hist.T, extent=[amp_bin_edges[0], amp_bin_edges[-1], phase_bin_edges[0], phase_bin_edges[-1]],
                       cmap=cmap, origin='lower', aspect='auto', vmin=0.1)  # vmin set to 0.1 to use 'under' color for empty bins
            plt.xlabel("Amplitude")
            plt.ylabel("Phase")
            plt.title("Fit Results")
            plt.colorbar()
            plt.grid()
            plt.tight_layout()

            # Get a list of the amp/phase bins sorted by counts
            bins = []
            for i in range(len(amp_bin_edges) - 1):
                for j in range(len(phase_bin_edges) - 1):
                    bins.append((i, j, hist[i, j]))
            bins.sort(key=lambda x: x[2], reverse=True)

            # For the largest 2 bins, compute chi-square and plot the chi-square landscape
            phases, amps = [], []
            for i in range(2):
                amp_bin, phase_bin, _ = bins[i]
                amp = (amp_bin_edges[amp_bin] + amp_bin_edges[amp_bin + 1]) / 2
                phase = (phase_bin_edges[phase_bin] + phase_bin_edges[phase_bin + 1]) / 2
                phases.append(phase)
                amps.append(amp)
            chi2_results = compute_chi_square(x, y_noisy, amplitude_range, np.array(phases), noise_std)
            plot_chi_square(amplitude_range, chi2_results, np.array(phases))
            plt.axvline(amps[0], color='r', label='Max Chi-Square 1')
            plt.axvline(amps[1], color='b', label='Max Chi-Square 2')
            plt.legend()

            # Plot the data with these two fits
            plt.figure(figsize=(8, 6))
            plot_data(x, y_noisy)
            for i in range(2):
                y_model = sinusoid_amp_phase(x, amps[i], phases[i])
                plt.plot(x, y_model, label=f'Amplitude: {amps[i]:.2f}, Phase: {phases[i]:.2f}')
            plt.legend()

            plt.show()
        #
        # # Plot data
        # plot_data(x, y_noisy, sinusoid(x, *popt))
        #
        # # Compute chi-square
        # chi2_results = compute_chi_square(x, y_noisy, amplitude_range, phase_shifts, noise_std)
        #
        # # Plot results
        # plot_chi_square(amplitude_range, chi2_results, phase_shifts)
    # Plot amp distribution
    plt.figure(figsize=(8, 6))
    plt.hist(amps, bins=50)
    plt.xlabel("Amplitude")
    plt.ylabel("Counts")
    plt.title("Amplitude Distribution")
    plt.grid()

    plt.figure(figsize=(8, 6))
    plt.hist(amps_0, bins=50)
    plt.xlabel("Amplitude")
    plt.ylabel("Counts")
    plt.title("Amplitude Distribution For 0")
    plt.grid()

    # Plot 2D scatter of amplitude vs std dev of amplitude
    plt.figure(figsize=(8, 6))
    plt.scatter(amps, amp_stds, alpha=0.5)
    plt.xlabel("Amplitude")
    plt.ylabel("Std Dev Amplitude")
    plt.title("Amplitude vs Std Dev Amplitude")
    plt.grid()


def sin_cos_func():
    # Parameters
    num_points = 6
    true_amplitude = 0.0
    frequency = 1.0 * 10000
    noise_std = 0.5
    amplitude_range = np.linspace(-1.5, 1.5, 50)
    n_fits = 50
    n_data = 1000
    fit_func = sinusoid_amp_phase_cos
    p0 = [0.0, 0.0]

    amps, amps_0, amp_stds = [], [], []
    for j in range(n_data):
        print(f'Data Set: {j + 1}')
        # Generate data
        x, y_noisy = generate_data(num_points, true_amplitude, frequency, noise_std)

        # Fit data multiple times for random initial parameters
        # p0 = [1.0]
        fit_amps, fit_phases, chi2_per_dofs = [], [], []
        for i in range(n_fits):
            rand_amp_sin = np.random.uniform(min(amplitude_range), max(amplitude_range))
            rand_amp_cos = np.random.uniform(min(amplitude_range), max(amplitude_range))
            p0[0], p0[1] = rand_amp_sin, rand_amp_cos
            popt, perr = fit_data(x, y_noisy, p0, bounds=None)
            chi2_per_dof_sin = chi_square(y_noisy, fit_func(x, *popt), noise_std) / (num_points - 2)
            phase = np.arctan(popt[1] / popt[0])
            amp = popt[0] / np.cos(phase)
            fit_amps.append(amp)
            fit_phases.append(phase)
            chi2_per_dofs.append(chi2_per_dof_sin)
        # Take the fit with the lowest chi-square
        min_chi2_index = np.argmin(chi2_per_dofs)
        popt = [fit_amps[min_chi2_index], fit_phases[min_chi2_index]]
        chi2_per_dof_sin = chi_square(y_noisy, fit_func(x, *popt), noise_std) / (num_points - 2)
        chi2_per_dof_0 = chi_square(y_noisy, np.zeros_like(y_noisy), noise_std) / num_points
        print(f'Chi-Square per DOF Sinusoid: {chi2_per_dof_sin:.2f}, Chi-Square per DOF 0: {chi2_per_dof_0:.2f}')

        if np.std(fit_amps) > 0:
            print(f'Std Dev Amplitude: {np.std(fit_amps):.2f}')
        amps.append(popt[0])
        amp_stds.append(np.std(fit_amps))
        if chi2_per_dof_0 < chi2_per_dof_sin:
            amps_0.append(0)
        else:
            amps_0.append(popt[0])

        # if np.std(fit_amps) > 0.2:
        if np.std(fit_amps) > 200.2:
            # Plot 2D histogram of fit results
            plt.figure(figsize=(8, 6))
            amp_bin_edges = np.linspace(-1.0, 1.0, 50)
            phase_bin_edges = np.linspace(-np.pi, np.pi, 50)
            hist, xedges, yedges = np.histogram2d(fit_amps, fit_phases, bins=[amp_bin_edges, phase_bin_edges])

            # Create a colormap with white for the lowest values
            cmap = plt.cm.jet
            cmap.set_under('white')

            plt.imshow(hist.T, extent=[amp_bin_edges[0], amp_bin_edges[-1], phase_bin_edges[0], phase_bin_edges[-1]],
                       cmap=cmap, origin='lower', aspect='auto', vmin=0.1)  # vmin set to 0.1 to use 'under' color for empty bins
            plt.xlabel("Amplitude")
            plt.ylabel("Phase")
            plt.title("Fit Results")
            plt.colorbar()
            plt.grid()
            plt.tight_layout()

            # Get a list of the amp/phase bins sorted by counts
            bins = []
            for i in range(len(amp_bin_edges) - 1):
                for j in range(len(phase_bin_edges) - 1):
                    bins.append((i, j, hist[i, j]))
            bins.sort(key=lambda x: x[2], reverse=True)

            # For the largest 2 bins, compute chi-square and plot the chi-square landscape
            phases, amps = [], []
            for i in range(2):
                amp_bin, phase_bin, _ = bins[i]
                amp = (amp_bin_edges[amp_bin] + amp_bin_edges[amp_bin + 1]) / 2
                phase = (phase_bin_edges[phase_bin] + phase_bin_edges[phase_bin + 1]) / 2
                phases.append(phase)
                amps.append(amp)
            chi2_results = compute_chi_square(x, y_noisy, amplitude_range, np.array(phases), noise_std)
            plot_chi_square(amplitude_range, chi2_results, np.array(phases))
            plt.axvline(amps[0], color='r', label='Max Chi-Square 1')
            plt.axvline(amps[1], color='b', label='Max Chi-Square 2')
            plt.legend()

            # Plot the data with these two fits
            plt.figure(figsize=(8, 6))
            plot_data(x, y_noisy)
            for i in range(2):
                y_model = sinusoid_amp_phase(x, amps[i], phases[i])
                plt.plot(x, y_model, label=f'Amplitude: {amps[i]:.2f}, Phase: {phases[i]:.2f}')
            plt.legend()

            plt.show()
        #
        # # Plot data
        # plot_data(x, y_noisy, sinusoid(x, *popt))
        #
        # # Compute chi-square
        # chi2_results = compute_chi_square(x, y_noisy, amplitude_range, phase_shifts, noise_std)
        #
        # # Plot results
        # plot_chi_square(amplitude_range, chi2_results, phase_shifts)
    # Plot amp distribution
    plt.figure(figsize=(8, 6))
    plt.hist(amps, bins=50)
    plt.xlabel("Amplitude")
    plt.ylabel("Counts")
    plt.title("Amplitude Distribution")
    plt.grid()

    plt.figure(figsize=(8, 6))
    plt.hist(amps_0, bins=50)
    plt.xlabel("Amplitude")
    plt.ylabel("Counts")
    plt.title("Amplitude Distribution For 0")
    plt.grid()

    # Plot 2D scatter of amplitude vs std dev of amplitude
    plt.figure(figsize=(8, 6))
    plt.scatter(amps, amp_stds, alpha=0.5)
    plt.xlabel("Amplitude")
    plt.ylabel("Std Dev Amplitude")
    plt.title("Amplitude vs Std Dev Amplitude")
    plt.grid()


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
    # popt, pcov = cf(sinusoid, x, y_data, p0=p0, bounds=([-np.inf, -2], [np.inf, 2]))
    # bounds = ([-np.inf], [np.inf]) if len(p0) == 1 else ([-np.inf, 0], [np.inf, np.pi / 2])
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


def plot_chi_square(amp_range, chi2_results, phase_shifts):
    """Plots chi-square as a function of amplitude for various phase shifts."""
    plt.figure(figsize=(8, 6))
    for phase in phase_shifts:
        plt.plot(amp_range, chi2_results[phase], label=f'Phase: {phase:.2f} rad')

    plt.xlabel("Amplitude")
    plt.ylabel("Chi-Square")
    plt.title("Chi-Square vs Amplitude for Different Phase Shifts")
    plt.legend()
    plt.grid()


if __name__ == '__main__':
    main()
