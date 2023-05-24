# Authors: Andre Wendlinger, andre.wendlinger@tum.de
#          Luca Nagel, luca.nagel@tum.de
#          Wolfgang Gottwald, wolfgang.gottwald@tum.de


# Institution: Technical University of Munich
# Date of last Edit (dd.mm.yyyy): 24.05.2023

import numpy as np
import scipy.constants as co
import matplotlib.pyplot as plt


def Get_Hist(data, bins=10):
    """
    Calculates Histogram from data for a number of bins.
    Parameters
    ----------
    data : np.array
        dataset from which to calculate the histogram
    bins : int
        number of bins

    Returns
    -------
    x_data, y_data, binsize
    """
    min_val = np.min(data)
    max_val = np.max(data)
    bin_size = (max_val - min_val) / bins
    x_data = np.linspace(min_val, max_val, bins)
    y_data = np.zeros_like(x_data)

    count = 0
    if len(data.shape) > 1:
        for i in np.arange(0, data.shape[0], 1):
            for j in np.arange(0, data.shape[1], 1):
                idx = (np.abs(x_data - data[i, j])).argmin()
                y_data[idx] += 1
    else:
        for j in np.arange(0, data.shape[0], 1):
            idx = (np.abs(x_data - data[j])).argmin()
            y_data[idx] += 1
    return x_data, y_data, bin_size


def flipangle_corr(T1_obs, flipangle, TR):
    """
    Corrects the observed flipangle for low flipangle experiments with a certain repetition time.
    Parameters
    ----------
    T1_obs : float
    observed T1 decay constant, usually around 50 for [1-13C]pyruvate
    flipangle : float
    Flipangle in radians
    TR : float
    Repetition time in seconds.

    Returns
    -------
    T1 : float
    corrected T1 decay constant
    """
    T1 = 1 / ((1 / T1_obs) + (np.log(np.cos(flipangle * np.pi / 180)) / TR))
    return T1

