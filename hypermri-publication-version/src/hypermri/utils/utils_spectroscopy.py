# Authors: Andre Wendlinger, andre.wendlinger@tum.de
#          Luca Nagel, luca.nagel@tum.de
#          Wolfgang Gottwald, wolfgang.gottwald@tum.de


# Institution: Technical University of Munich
# Date of last Edit (dd.mm.yyyy): 24.05.2023

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ipywidgets as widgets


def lorentzian(x, FWHM, center_position, peak_height):
    return peak_height * FWHM**2 / ((FWHM**2 + (2 * x - 2 * center_position) ** 2))


def norm(x, a=0, b=1):
    """Norm an array to interval [a,b] with a<b. Default range is [0,1].

    To convert spectroscopy x-axis from ppm to Hz:

        x_Hz = norm(x_ppm - 4.7, -1, 1) * (scan.method['PVM_SpecSWH'] / 2
    """
    assert a < b

    min_x = np.min(x)
    max_x = np.max(x)

    normed_x_01 = (x - min_x) / (max_x - min_x)  # normed array from 0 to 1
    normed_x_custom = (b - a) * normed_x_01 + a  # normed array within [a,b]

    return normed_x_custom


def get_ppm_axis(scan, cut_off=70):
    """Get frequency axis of given spectroscopy scan in units of ppm.

    Returns ppm axis for spectroscopic measurements given a certain cut_off
    value at which fid will be cut off. The default cut off value of 70 points
    is usually sufficient as there is no signal left.

    Similar to get_Hz_axis() function.

    Parameters
    ----------
    cut_off : int
        Default value is 70. After 'cut_off' points the signal is truncated.

    Returns
    -------
    ppm_axis : np.ndarray
        Frequency axis of measurement in units of ppm.
    """
    center_ppm = float(scan.method["PVM_FrqWorkPpm"][0])
    BW_ppm = float(scan.method["PVM_SpecSW"])
    acq_points = int(scan.method["PVM_SpecMatrix"])

    ppm_axis = np.linspace(
        center_ppm - BW_ppm / 2, center_ppm + BW_ppm / 2, acq_points - cut_off
    )

    return ppm_axis


def get_Hz_axis(scan, cut_off=70):
    """Get frequency axis of given spectroscopy scan in units of Hz.

    Returns Hz axis for spectroscopic measurements given a certain cut_off
    value at which fid will be cut off. The default cut off value of 70 points
    is usually sufficient as there is no signal left.

    Similair to get_Hz_axis() function.

    Parameters
    ----------
    cut_off : int
        Default value is 70. After 'cut_off' points the signal is truncated.

    Returns
    -------
    Hz_axis : np.ndarray
        Frequency axis of measurement in units of Hz.
    """
    BW_Hz = float(scan.method["PVM_SpecSWH"])
    acq_points = int(scan.method["PVM_SpecMatrix"])

    Hz_axis = np.linspace(-1, 1, acq_points - cut_off) * (BW_Hz / 2)

    return Hz_axis


def norm_spectrum_to_snr(spec, bg_indices=[0, 100]):
    """
    Norms input spectrum to background region (default is first 100 entries, can be changed)
    Parameters
    ----------
    spec: array
        shape: (Repetitions, Spectral_acquisition_points), contains spectra for a given method.
    bg_indices: list, optional
    two entry list with indices of background region, default is [0,100]
    Returns
    -------
    normed_spec: array
    """
    normed_spec = np.zeros_like(spec)
    for n in range(0, spec.shape[0], 1):
        normed_spec[n, :] = (
            spec[n, :] - np.mean(spec[n, bg_indices[0] : bg_indices[1]])
        ) / np.std(spec[n, bg_indices[0] : bg_indices[1]])
    return normed_spec


def fit_spectrum(
    ppm_axis,
    spectrum,
    peak_positions,
    SNR_cutoff=1,
    plot=False,
    norm_to_snr_before_fit=False,
    bg_region_first_spec=[1200, 1800],
):
    """
    Fits lorentzian to a spectrum at the desired peak locations
    Parameters
    ----------
    ppm_axis: numpy array
        The ppm-scale of the measurements to be examined.
    spectrum: numpy array
        N-dimensional spectrum to be fitted.
    peak_positions: list
        Contains the peak positions where fits should occur in ppm.
    SNR_cutoff: float
        SNR value below which peaks should not be fitted, default is 5.
    plot: bool
        Select if plotting for checking is needed
    norm_to_snr_before_fit: bool
        Select if you want to norm the spectra to a background region of the first repetition before fitting.
    bg_region_first_spec: list
        indices of the background region from which the snr is calculated.

    Returns
    -------
    peak_coeff: numpy array
        coefficients for a 3 parametric lorentzian fit model, shape: (NRepetitions, Number_of_peaks, 3)
        peak_coeff[0]: peak FWHM
        peak_coeff[1]: peak position (ppm)
        peak_coeff[2]: peak SNR

    peak_errors: numpy array
        errors for fit coefficients calculated from the covariance matrix, shape: (NRepetitions, Number_of_peaks, 3)
    """

    def find_range(axis, ppm):
        return np.argmin(np.abs(axis - ppm))

    # Interpolating the ppm axis
    ppm_itp = np.linspace(np.min(ppm_axis), np.max(ppm_axis), 10000)
    # Norming to SNR
    # use first repetition where there should be no signal as reference region for SNR

    if norm_to_snr_before_fit:
        # norm to background region of first repetition
        spec_norm = np.zeros_like(spectrum)
        for n in range(0, spectrum.shape[0], 1):
            spec_norm[n, :] = (
                spectrum[n, :]
                - np.mean(
                    spectrum[0, bg_region_first_spec[0] : bg_region_first_spec[1]]
                )
            ) / np.std(spectrum[0, bg_region_first_spec[0] : bg_region_first_spec[1]])
    else:
        # just baseline correction
        spec_norm = np.zeros_like(spectrum)
        for n in range(0, spectrum.shape[0], 1):
            spec_norm[n, :] = spectrum[n, :] - np.mean(
                spectrum[0, bg_region_first_spec[0] : bg_region_first_spec[1]]
            )
    # defining in with region we fit
    width = 1
    # number of reps
    NR = spectrum.shape[0]
    N_peaks = len(peak_positions)
    # defining region in which we will fit

    peak_coeff = np.zeros((NR, N_peaks, 3))
    peak_covariance = np.zeros((NR, N_peaks, 3, 3))
    peak_errors = np.zeros((NR, N_peaks, 3))

    for repetition in range(NR):
        for peak_number, peak_center in enumerate(peak_positions):
            peak_roi = [
                find_range(ppm_axis, peak_center - width),
                find_range(ppm_axis, peak_center + width),
            ]
            try:
                (
                    peak_coeff[repetition, peak_number],
                    peak_covariance[repetition, peak_number],
                ) = curve_fit(
                    lorentzian,
                    ppm_axis[peak_roi[0] : peak_roi[1]],
                    spec_norm[repetition, peak_roi[0] : peak_roi[1]],
                    bounds=(
                        [
                            0.01,
                            peak_center - width / 2.0,
                            np.min(spec_norm[repetition]),
                        ],
                        [5, peak_center + width / 2.0, np.max(spec_norm[repetition])],
                    ),
                )
                peak_errors[repetition, peak_number] = np.sqrt(
                    np.diag(peak_covariance[repetition, peak_number])
                )

            except RuntimeError:
                #print(
                #    "Repetition " + str(repetition),
                #    " - Peak at",
                #    str(peak_center),
                #    " could not be fitted, possibly no signal",
                #)
                peak_coeff[repetition, peak_number] = None
                peak_covariance[repetition, peak_number] = None
                peak_errors[repetition, peak_number] = None
            # clean up badly fitted peaks
    for peak in range(peak_coeff.shape[1]):
        for repetition in range(peak_coeff.shape[0]):
            peak_snr = peak_coeff[repetition, peak][2]
            peak_snr_error = peak_errors[repetition, peak][2]
            if peak_snr > SNR_cutoff:
                # peak needs to have an SNR greater than a certain value
                pass
            else:
                peak_coeff[repetition, peak] = [0, 0, 0]

    if plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        @widgets.interact(repetition=(0, NR - 1, 1))
        def update(repetition=0):
            [l.remove() for l in ax[0].lines]
            [l.remove() for l in ax[0].lines]

            # Plotting for QA
            combined_fits = 0
            for peak_number, peak in enumerate(peak_positions):
                combined_fits += lorentzian(
                    ppm_itp,
                    peak_coeff[repetition, peak_number][0],
                    peak_coeff[repetition, peak_number][1],
                    peak_coeff[repetition, peak_number][2],
                )
            ax[0].plot(
                ppm_axis,
                spec_norm[repetition, :],
                linewidth="0.5",
                color="r",
                label="Data",
            )
            ax[0].plot(
                ppm_itp,
                combined_fits,
                linestyle="dashed",
                color="k",
                label="Lorentzian fit",
            )

            ax[0].set_title("Repetition " + str(repetition))
            for peak_number, peak in enumerate(peak_positions):
                print(str(peak), peak_coeff[repetition, peak_number])
            ax[0].set_xlim([190, 160])
            ax[0].set_ylim(
                [np.min(spec_norm[repetition, :]), np.max(spec_norm[repetition, :])]
            )
            ax[0].set_ylabel("SNR")
            ax[0].set_xlabel(r"$\sigma$[ppm]")

            ax[0].legend()

            [l.remove() for l in ax[1].lines]
            [l.remove() for l in ax[1].lines]

            # Plotting for QA

            for peak_number, peak in enumerate(peak_positions):
                ax[1].plot(
                    ppm_itp,
                    lorentzian(
                        ppm_itp,
                        peak_coeff[repetition, peak_number][0],
                        peak_coeff[repetition, peak_number][1],
                        peak_coeff[repetition, peak_number][2],
                    )
                    - 20 * peak_number,
                    color="C" + str(peak_number),
                    label=str(peak) + " ppm",
                )
            ax[1].set_ylabel("I [a.u.]")
            ax[1].set_yticks([])
            ax[1].set_xlabel(r"$\sigma$[ppm]")
            ax[1].set_title("Repetition " + str(repetition))
            ax[1].legend()
            ax[1].set_xlim([190, 160])

        if norm_to_snr_before_fit:
            # plotting of background region for snr calculation
            fig_bg, bg_axis = plt.subplots(1)
            bg_axis.set_title("Background region - spectrum 0")
            points = np.linspace(0, spectrum[0, :].shape[0], spectrum[0, :].shape[0])
            bg_axis.plot(points, spectrum[0, :])

            bg_axis.fill_between(
                [points[bg_region_first_spec[0]], points[bg_region_first_spec[1]]],
                np.min(spectrum[0]),
                np.max(spectrum[0]),
                alpha=0.3,
                color="C1",
            )
        else:
            pass
    else:
        pass

    return peak_coeff, peak_errors


def integrate_fitted_spectrum(
    experiment_instance,
    ppm_axis,
    spectrum,
    peak_positions,
    peak_coeff,
    plot=False,
    plot_title=None,
    savepath=None,
):
    """
    Integrates and displays a time curve for 1-D spectral data that was fitted using fit_spectrum
    Parameters
    ----------
    experiment_instance: BrukerExp instance which contains meta data of the experiment that is being fitted
        This can be a NSPECT type for example.
    ppm_axis: numpy array
        The ppm-scale of the measurements to be examined.
    spectrum: numpy array
        N-dimensional spectrum to be fitted.
    peak_positions: list
        Contains the peak positions where fits where done.
    peak_coeff: numpy array
        coefficients for a 3 parametric lorentzian fit model, shape: (NRepetitions, Number_of_peaks, 3)
    plot: bool
        Select if plot for checking is wanted
    plot_title: str
        title of plot
    Returns
    -------
    peak_integrals: numpy array
        shape: (N_repetitions, N_peaks, 1)
    """
    # interpolate ppm-axis
    # number of reps
    NR = spectrum.shape[0]
    N_peaks = len(peak_positions)
    TR = experiment_instance.method["PVM_RepetitionTime"]
    time_scale = np.arange(0, TR * NR, TR) / 1000
    ppm_itp = np.linspace(np.min(ppm_axis), np.max(ppm_axis), 10000)
    peak_integrals = np.zeros((NR, N_peaks))
    for repetition in range(NR):
        for peak_number, peak in enumerate(peak_positions):
            # integrate the fits
            peak_integrals[repetition, peak_number] = np.sum(
                np.abs(
                    lorentzian(
                        ppm_itp,
                        peak_coeff[repetition, peak_number][0],
                        peak_coeff[repetition, peak_number][1],
                        peak_coeff[repetition, peak_number][2],
                    )
                )
            )
    peak_integrals = peak_integrals
    if plot:
        fig, ax = plt.subplots(1)
        for n in range(len(peak_positions)):
            ax.plot(
                time_scale, peak_integrals[:, n], label=str(peak_positions[n]) + " ppm"
            )
        ax.set_xlabel("Time [s] ")
        ax.set_ylabel("I [a.u.]")
        ax.legend()
        ax.set_title(plot_title)
    if savepath:
        plt.savefig(savepath + plot_title + "_timecurve.png")
    return peak_integrals


def calculate_pyruvate_to_lactate_auc(lactate_timecurve, pyruvate_timecurve):
    """
    Calculates pyruvate to lactate area under the curve ratios for a given experiment.
    Parameters
    ----------
    lactate_timecurve: numpy array
    pyruvate_timecurve: numpy array

    Returns
    -------
    lac_pyr: float
        lactate/pyruvate ratio
    """
    pyruvate_auc = np.sum(pyruvate_timecurve)
    lactate_auc = np.sum(lactate_timecurve)
    lac_pyr = lactate_auc / pyruvate_auc
    #print("Lactate / Pyruvate ratio of AUCs = ", lac_pyr)
    return lac_pyr
