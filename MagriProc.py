"""
Author: Wolfgang Gottwald, wolfgang.gottwald@tum.de
# Institution: Technical University of Munich
# Date of last Edit (dd.mm.yyyy): 24.05.2023

This package is used for analysis of 1T Tabletop NMR data from a Magritek Spinsolve NMR Spectrometer.
It uses loading functions from the following package:
https://github.com/bennomeier/pyNMR which needed to be copied since they are not fully implemented there.
We asked for permission at https://github.com/bennomeier/pyNMR/issues/30 and included the License here in this repository under
ExternalLicenses/LICENSE_pyNMR.
"""

import os
import struct
import numpy as np
import scipy.constants as co
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
import ipywidgets as widgets

class MagriExp:
    def __init__(self, path, folder_name):
        """
        Parameters
        ----------
        path: Path to where the measurements are saved in the respective folders for more scans
        folder_name: name of folder as int or str depending if its numbered (expert folders) or
                        not

        """
        self.path = path
        self.folder_name = folder_name
        self.fid = self.__load()[0]
        self.parDict = self.__load()[1]
        self.frequency_axis = self.__load()[2]
        self.ppm_axis = self.__load()[3]
        self.spec = self.get_spec()
        # check if a manual offset of ppm axis is needed (e.g. if sequence was modified)
        #self.shift_ppm_axis()
        self.averages = float(self.parDict['nrScans'])
        try:
            # Spinsolve expert
            self.TR = float(self.parDict['exptInterval'])
            self.flipangle = float(self.parDict['pulseLength13CNominal'])
            self.rxgain = float(self.parDict['recGain'])
        except:
            # Spinsolve normal
            self.TR = float(self.parDict['repTime']) / 1000  # in ms
            self.flipangle = float(self.parDict['pulseAngle'])
            self.rxgain = float(self.parDict['rxGain'])

    def __load(self):
        """
        Loads data.1d and acqu.par files. Taken and adjusted from pyNMR (https://github.com/bennomeier/pyNMR)
        Returns
        -------
        fid_data: np.array
        parDict: dict
        frequency_axis: np.array
        ppm_axis: np.array
        """
        fid_data = []
        parDict = {}
        if os.path.isfile(self.path + str(self.folder_name) + "/data.1d"):
            f = open(self.path + str(self.folder_name) + "/data.1d", "rb")
            f.seek(12)
            # get this information out of the acqu file.
            f.seek(16)
            sizeTD2 = struct.unpack('<i', f.read(4))[0]
            f.seek(32)
            # t just contains floats with the time
            t = struct.unpack('<' + 'f' * sizeTD2, f.read(4 * sizeTD2))
            data1 = struct.unpack('<' + 'f' * sizeTD2 * 2, f.read(4 * sizeTD2 * 2))
            realPart = np.array(data1[::2])
            imagPart = np.array(data1[1::2])

            fid_data.append(realPart + 1j * imagPart)

            fid_data = np.array(fid_data)[0]
            fid_points = fid_data.shape[0]

        else:
            raise FileNotFoundError(str(self.path + str(self.folder_name) + "/data.1d") + ' was not found')
        if os.path.isfile(self.path + str(self.folder_name) + "/acqu.par"):

            f_acqu = open(self.path + str(self.folder_name) + "/acqu.par", "r")

            count = 0
            while True:
                count += 1
                line = f_acqu.readline().strip()
                if "=" in line:
                    line = line.split("=")


                elif len(line) == 0 or count > 3000:
                    # print("Ended dreading acqus file at line ", count)
                    break
                else:
                    next
                if len(line) > 1:
                    parDict[line[0].strip()] = line[1].strip()

            freq_bw = int(1. / (float(parDict["dwellTime"]) * 1e-6))

            # frequency axis is defined by lowest frequency plus bandwidth
            lowest_freq = float(parDict['lowestFrequency'])
            # for Spinsolve Expert measurements the lowest freq parameter is wrong by a factor
            # of 100, this seems to be a software issue and will be investigated with Magritek
            # however for the time being we will divide it by 100 to have the correct scale again
            if lowest_freq < -2e3:
                lowest_freq = lowest_freq / 100
            else:
                pass
            carrier_freq = float(parDict['b1Freq'])
            acqpts = int(parDict['nrPnts'])
            if acqpts != fid_points:
                # for Spinsolve (non expert) scans, the acquisiton points
                # value in the acqp file is wrong somehow
                # therefore we take the number of points the fid has
                # maybe zero filling or interpoltion are the issue here
                acqpts = fid_points
            else:
                pass
            # making the ppm and frequency axis
            # FIXME this does not produce a correct result for sequences in Spinsolve Expert with lower dwell time
            frequency_axis = np.linspace(lowest_freq, lowest_freq + freq_bw, acqpts)
            ppm_axis = np.linspace((lowest_freq + freq_bw) / carrier_freq, lowest_freq / carrier_freq, acqpts)
        else:
            frequency_axis = []
            ppm_axis = []
            raise FileNotFoundError(str(self.path + str(self.folder_name) + "/acqu.par") + ' was not found')

        return fid_data, parDict, frequency_axis, ppm_axis

    def get_spec(self):
        """
        Fourier transforms the fid and returns a spectrum.

        Returns
        -------
        spec: list
        """
        spec = np.abs(np.fft.fftshift(np.fft.fft(self.fid)))
        return spec

    def norm_spec(self, bg=[300, 600]):
        """
        Norms spectrum to background noise region.
        Parameters
        -------
        bg: array, optional
            background region indices,default is [300,600]

        Returns
        -------
        normed_spec
        """
        spec = self.spec

        normed = (spec - np.mean(spec[bg[0]:bg[1]])) / np.std(spec[bg[0]:bg[1]])
        return normed


class MagriExps:
    """
    Class for multiple scans within a experiment, i.e. repetitions.
    """
    def __init__(self, path, folders):
        self.path = path
        self.folders = folders
        self.exp_list = self.__get_exp_list()
        self.fids, self.specs, self.ppm_axis, self.frequency_axis = self.load_multiple_scans()

        if type(folders) in [int, str]:
            # for single measurement data
            self.parDict = self.exp_list.parDict
            self.TR = self.exp_list.TR
            self.number_of_spectra = 1
            self.averages = self.exp_list.averages
            self.flipangle = self.exp_list.flipangle
            self.rxgain = self.exp_list.rxgain
        else:
            self.parDict = self.exp_list[0].parDict
            self.TR = self.exp_list[0].TR
            self.number_of_spectra = len(folders)
            self.averages = self.exp_list[0].averages
            self.flipangle = self.exp_list[0].flipangle
            self.rxgain = self.exp_list[0].rxgain

    def __get_exp_list(self):
        exp_list = []
        # allow single measurements to be loaded in
        if type(self.folders) in [int, str]:
            exp_list = MagriExp(self.path, self.folders)
        else:
            # if we have multiple scans append the instances to list
            for folder_name in self.folders:
                exp_list.append(MagriExp(self.path, folder_name))
        return exp_list

    def load_multiple_scans(self):
        """
        Loads multiple folders and returns the fids, spectra and ppm as well as frequency axes.
        Used by __init__
        Returns
        -------
        fids: numpy array
            contains the fid for each spectrum
        specs: numpy array
            contains the fouriertransformed fids
        ppm_axis: numpy array
            ppm-scale used for plotting
        frequency_axis: numpy array
            Hz-scale used for plotting
        """
        if type(self.folders) in [int, str]:
            experiment = self.exp_list
            fids, specs, ppm_axis, frequency_axis = experiment.fid, experiment.spec, experiment.ppm_axis, experiment.frequency_axis
        else:
            counter = 0
            experiment = self.exp_list[counter]
            ppm_axis = experiment.ppm_axis
            frequency_axis = experiment.frequency_axis
            fids, specs = np.zeros((len(self.folders), int(experiment.parDict['nrPnts'])), dtype=np.complex128), \
                          np.zeros((len(self.folders), int(experiment.parDict['nrPnts'])))
            while counter < len(self.folders):
                experiment = self.exp_list[counter]
                fids[counter, :] = experiment.fid
                specs[counter, :] = experiment.spec
                counter += 1
        return fids, specs, ppm_axis, frequency_axis

    def average_thermal_spectra(self, lb=0):
        """
        Averages a stack of fids and optionally applies linebroadening to the averaged fid.
        Parameters
        ----------
        lb : float, optional
            Applied linebroadening in Hz to averaged fid.

        Returns
        -------
        spec: np.array
            Fouriertransform of the averaged fid.
        """
        avg_fids = np.mean(self.fids, axis=0)
        time_axis = np.linspace(0, float(self.parDict['acqTime']), int(self.parDict['nrPnts'])) / 1000
        lb_fac = np.exp(-2 * np.pi * lb * time_axis)
        spec = np.abs(np.fft.fftshift(np.fft.fft(avg_fids * lb_fac)))
        return spec

    def phase_real_spectrum(self,average_spectra_flag):
        """
        Interactively allows phasing of spectra.
        Parameters
        ----------
        average_spectra_flag: bool, if True the input are thermal low SNR spectra that need to be averaged.
        """

        if average_spectra_flag is True:
            # these are thermal spectra that need to be averaged
            fid = np.mean(self.fids,axis=0)
        else:
            if len(self.fids.shape)>1:
                # in case we have hyper spectra take first
                fid = self.fids[0,:]
            else:
                # if we only have one spectrum take that
                fid = self.fids

        spec_complex  =  np.fft.fftshift(np.fft.fft(fid))
        # perform a baseline correction
        spec_base_line_corr = spec_complex-np.mean(spec_complex[5000:8500])
        # phase the real spectrum
        Integrals_th = []
        phases = np.linspace(0, 360, 1000)
        for phase in phases:
            itgl = np.sum(np.real(spec_base_line_corr * np.exp(1j * (phase * np.pi) / 180.0)))
            Integrals_th.append(itgl)
        initial_guess_phase = phases[np.argmin(np.abs(Integrals_th - np.max(Integrals_th)))]

        fig,ax=plt.subplots(1,figsize=(12,4),tight_layout=True)

        line_real, = ax.plot(np.real(spec_base_line_corr* np.exp(1j * (initial_guess_phase * np.pi) / 180.0)),label='Real-phased',color='k')
        line_abs,= ax.plot(np.abs(spec_base_line_corr)-np.mean(np.abs(spec_base_line_corr[8000:9000])),label='Magnitude',color='r')
        ax.hlines(0,0,len(spec_base_line_corr),linestyles='dashed',alpha=0.3,color='b',label='Baseline')
        ax.fill_between([8000, 9000],np.min(np.real(spec_base_line_corr)),np.max(np.real(spec_base_line_corr)), alpha=0.3, color='C2', label='Background')
        ax.set_xlabel('Points')
        ax.set_ylabel('I [a.u.]')
        ax.set_title('Phased with '+str(np.round(phase,1))+' °')
        ax.legend()

        @widgets.interact(phase=(0,360,0.1))
        def update(phase):
            line_real.set_ydata(np.real(spec_base_line_corr* np.exp(1j * (phase * np.pi) / 180.0)))
            line_abs.set_ydata(np.abs(spec_base_line_corr)-np.mean(np.abs(spec_base_line_corr[8000:9000])))
            ax.set_title('Phased with '+str(np.round(phase,1))+' °')





def Fit_T1(dataset,savepath,sample_name,first_spectrum=0,linebroadening=0,integration_width=5,
           guessed_T1=70,select_peak_ppm=False,plot=False,ppm_axis_offset = 621):
    """
    Fits mono-exponential to a hyperpolarized data set.
    Parameters
    ----------
    dataset: MagriExps instance containing data
    savepath: str/bool
        if not False, plot is saved to the desired directory specified by the path.
    sample_name: str
        Identificator for sample
    first_spectrum: int, optional
        in case we want to fit only to later spectra, by default 0.
    linebroadening: float, optional
        Applied linebroadening in Hz, by default 0.
    integration_width: int, optional
        Desired integration width in ppm around the largest peak, by default 0.
    guessed_T1: float, optional
        Guess for T1 value to make fit converge easier, by default 50.
    select_peak_ppm: float, optional
        Ppm value of a selected peak that should be integrated, by default False which then takes the largest peak.
    plot: bool, optional
        Select True if plots are wanted, by default False.
    ppm_axis_offset: float, optional
        For some measurements the ppm scale the Magritek returns is wrong. In this case enter the offset (i.e. where the peak
        is in the plot against where it should be) here. The default is 621 which sets the pyruvate peak in most measurements
        to 171ppm.
    Returns
    -------
    T1: float
        decay constant
    """
    FA_hyper = dataset.flipangle * np.pi / 180.
    # get the number of spectra for hyper measurement
    Nspec_hyper = dataset.number_of_spectra
    # Repetition time
    TR_hyper = float(dataset.TR)
    ppm_axis_hyper = dataset.ppm_axis-ppm_axis_offset
    spectra = []
    # apply linebroadening

    time_axis = np.linspace(0, float(dataset.parDict['acqTime']), int(dataset.parDict['nrPnts'])) / 1000
    lb_fac = np.exp(-2 * np.pi * linebroadening * time_axis)
    for spectrum in range(Nspec_hyper):
        spec = np.abs(np.fft.fftshift(np.fft.fft(dataset.fids[spectrum, :] * lb_fac)))
        spectra.append(spec)
    spectra = np.array(spectra)
    # peak center

    if select_peak_ppm:
        center_ppm_hyper = select_peak_ppm

    else:
        # find largest peak
        center_hyper = np.squeeze(np.where(spectra - np.max(spectra) == 0))[1]
        center_ppm_hyper = ppm_axis_hyper[center_hyper]


    lower_bound_integration_ppm_hyper = np.abs(ppm_axis_hyper - (center_ppm_hyper - integration_width))
    lower_bound_integration_index_hyper = np.argmin(
        lower_bound_integration_ppm_hyper - np.min(lower_bound_integration_ppm_hyper))
    upper_bound_integration_ppm_hyper = np.abs(ppm_axis_hyper - (center_ppm_hyper + integration_width))
    upper_bound_integration_index_hyper = np.argmin(
        upper_bound_integration_ppm_hyper - np.min(upper_bound_integration_ppm_hyper))
    # from this we calculate the integrated peak region
    # sorted so that lower index is first
    integrated_peak_roi_hyper = [lower_bound_integration_index_hyper, upper_bound_integration_index_hyper]
    integrated_peak_roi_hyper.sort()

    SNR_hyper_arr = [np.sum(spectra[n, integrated_peak_roi_hyper[0]:integrated_peak_roi_hyper[1]]) for n in
                     range(0, Nspec_hyper, 1)]

    # fit exponential to hyper SNR to backcalculate
    def exponential(x, M, T1):
        return M * np.exp(-x / T1)

    # define the time axis during which the scans took place
    hyp_time_axis = np.arange(0, TR_hyper * Nspec_hyper, TR_hyper)
    #dropping first spectra if needed
    hyp_time_axis = hyp_time_axis[first_spectrum:]
    SNR_hyper_arr = SNR_hyper_arr[first_spectrum:]

    coeff, err = curve_fit(exponential, hyp_time_axis, SNR_hyper_arr, p0=(np.max(SNR_hyper_arr), guessed_T1))

    # flipangle correct for time outside bore
    flipangle_corr_T1 = flipangle_corr(coeff[1], FA_hyper, TR_hyper)
    error_t1 = np.sqrt(np.diag(err))[1]

    print('Magritek measurement -- T1=',np.round(flipangle_corr_T1,1),'plus minus',np.round(error_t1,1),' s')
    if plot:

        fig,ax=plt.subplots(1,2,tight_layout=True,figsize=(12,4))
        ax[0].scatter(hyp_time_axis, SNR_hyper_arr, label='Data points')

        ax[0].plot(hyp_time_axis, exponential(hyp_time_axis, coeff[0], coeff[1]),
                   label='Fit - T1=' + str(np.round(coeff[1], 1)) + 's')
        ax[0].plot(hyp_time_axis,
                   exponential(hyp_time_axis, coeff[0], flipangle_corr_T1),
                   label='Corrected Fit - T1=' + str(np.round(flipangle_corr_T1, 1)) + 's')
        ax[0].legend()
        ax[0].set_ylabel('I [a.u.]')
        ax[0].set_xlabel('Time since start of experiment [s]')
        ax[0].set_title('T1(corr) = '+str(np.round(flipangle_corr_T1, 1))+r'$\pm$'+str(np.round(error_t1,1))+' s')

        ax[1].plot(ppm_axis_hyper,spectra[0,:])
        ax[1].fill_between(
                [ppm_axis_hyper[integrated_peak_roi_hyper[0]], ppm_axis_hyper[integrated_peak_roi_hyper[1]]],
                np.min(spectra), np.max(spectra), alpha=0.3, color='C1', label='Peak integration')
        ax[1].legend()
        ax[1].set_xlabel(r'$\sigma$[ppm]')
        ax[1].set_xlim([np.max(ppm_axis_hyper),np.min(ppm_axis_hyper)])
        ax[1].set_ylabel('I [a.u.]')

        if savepath:
            plt.savefig(savepath+'Sample_'+str(sample_name)+'_t1_measurement_1T.png')
    else:
        pass

    return flipangle_corr_T1,error_t1

# Helper functions of the package
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
    T1 = 1 / ((1 / T1_obs) + (np.log(np.cos(flipangle)) / TR))
    return T1


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

def calculate_polarization_level(thermal_data, hyper_data, time_to_diss, bg_region_hyper=[250,400],
                      bg_region_thermal=[250,400], molarity_hyper=0.08, molarity_thermal=0.08,
                      T1_for_backcalculation = False,
                      linebroadening=0, integration_width_hyper=5, integration_width_thermal=5,select_peak_ppm_thermal = False,
                      select_peak_ppm_hyper=False, first_spec=0,thermal_phase_input=False,hyper_phase_input=False, B_field=1, Temp=28.5, print_output=False, plot=False):
    """
    Calculates the polarization level by comparing a thermal dataset with a hyperpolarized one.

    Parameters
    ----------
    thermal_data : MagriExps instance
        Contains thermal spectra (.specs) and meta data (.acqp)
    hyper_data : MagriExps instance
        Contains hyperpolarized spectra (.specs) and meta data (.acqp)
    time_to_diss : float
        Time from experiment to dissolution
    bg_region_hyper : list, optional
        ppm values where background region is taken from, by default [250,400]
    bg_region_thermal : list, optional
        ppm values where background region is taken from, by default [250,400]
    molarity_hyper : float, optional
        Molarity of hyperpolarized sample in mols / l, for our setup this is
        80mM, by default 0.08
    molarity_thermal : float, optional
        Molarity of thermal sample, by default 0.08
    T1_for_backcalculation : bool/float, optional
        Gives us the option to use an externally known T1 for the decay outside the bore in seconds.
        If False, it uses the calculated/flipangle corrected T1 inside the bore.
    linebroadening : float, optional
        Linebroadening applied to both spectra before integration, default is 0 Hz.
    integration_width_hyper : float, optional
        Integration width around peak of hyper spectra in ppm, default is 3.
    integration_width_thermal : float, optional
        Integration width around peak of thermal spectrum in ppm, default is 3.
    first_spec : int, optional
        First repetition that is used, by default 0.
    B_field : int, optional
        Magnetic field in spectrometer in Tesla, by default 1
    Temp : int, optional
        Temperature in the bore in degree Celsius, by default 28.5
    print_output : bool,optional
        Prints results into notebook, by default False
    plot: bool, optional
        Plots results, by default False.
    Returns
    -------
    Polarization_level : float
        Polarization level in hyperpolarized state at time_to_diss in percent.
    Polarization_level_at_first_spec: float
        Polarization level in hyperpolarized state at first spectrum in percent.
    SNR_thermal : float
        Thermal SNR from measurement
    SNR_hyper : float
        Hyperpolarized SNR backcalculated to time_to_diss, corrected for flipangle.
    Pol_lvl_thermal : float
        Thermal polarization level (NOT IN PERCENT)
    enhancement_factor: float
        enhancement from thermal to hyper state
    flipangle_corr_T1: float
        T1 flip angle corrected in s
    """
    # Step 1
    # calculate thermal polarization level according to Boltzmann law
    Pol_lvl_thermal = np.tanh(co.hbar * 67.2828 * 1e6 * B_field / (2 * co.k * (273.15 + Temp)))

    '------------------------------------------------------------------------------------------------------------'
    '------------------------------------------------------------------------------------------------------------'
    '------------------------------------------------------------------------------------------------------------'
    # Step 2: calculate hyperpolarized SNR dependent on time
    # Norm first selected spectrum thats useable to background noise level
    # and peform baseline correction by subtracting the mean of the background region
    # Integrate the peak to obtain SNR
    # To get the decay constant T1, integrate the raw signal array that has not been normed to background noise
    # fit it to exponential decay function
    # correct the T1 decay constant through the flipangle and TR of the sequence used to monitor the
    # hyperpolarized decay

    # convert input flip angle to radians
    FA_hyper = hyper_data.flipangle * np.pi / 180.
    # get the number of spectra for hyper measurement
    Nspec_hyper = hyper_data.number_of_spectra
    # Repetition time
    TR_hyper = float(hyper_data.TR)
    TR_thermal= float(thermal_data.TR)
    ppm_axis_hyper = hyper_data.ppm_axis
    lower_bound_bg_hyper = np.abs(ppm_axis_hyper - bg_region_hyper[0])
    lower_bound_bg_index_hyper = np.argmin(lower_bound_bg_hyper - np.min(lower_bound_bg_hyper))
    upper_bound_bg_hyper = np.abs(ppm_axis_hyper - bg_region_hyper[1])
    upper_bound_bg_index_hyper = np.argmin(upper_bound_bg_hyper - np.min(upper_bound_bg_hyper))
    bg_region_hyper_indices = [lower_bound_bg_index_hyper, upper_bound_bg_index_hyper]
    bg_region_hyper_indices.sort()

    # apply linebroadening in case its wanted
    time_axis = np.linspace(0, float(hyper_data.parDict['acqTime']), int(hyper_data.parDict['nrPnts'])) / 1000
    lb_fac = np.exp(-2 * np.pi * linebroadening * time_axis)
    # take the real part of the first selected spectrum and phase it
    hyper_spec_1_complex  =  np.fft.fftshift(np.fft.fft(hyper_data.fids[first_spec, :]*lb_fac))
    # perform a baseline correction
    hyper_spec_1_complex_baseline_corr = hyper_spec_1_complex-np.mean(hyper_spec_1_complex[bg_region_hyper_indices[0]:bg_region_hyper_indices[1]])
    # phase the real spectrum
    Integrals_hyp = []
    phases = np.linspace(0, 360, 1000)
    for phase in phases:
        itgl = np.max(np.real(hyper_spec_1_complex_baseline_corr * np.exp(1j * (phase * np.pi) / 180.0)))
        Integrals_hyp.append(itgl)


    # take the real part of the spectrum
    if hyper_phase_input is False:
        final_phase_hyper = phases[np.argmin(np.abs(Integrals_hyp - np.max(Integrals_hyp)))]
    else:
        final_phase_hyper = hyper_phase_input
    first_hyper_spec = np.real(hyper_spec_1_complex_baseline_corr * np.exp(1j * (final_phase_hyper * np.pi) / 180.0))
    # norm to background region
    hyper_normed = (first_hyper_spec-np.mean(first_hyper_spec[bg_region_hyper_indices[0]:bg_region_hyper_indices[1]]))/np.std(first_hyper_spec[bg_region_hyper_indices[0]:bg_region_hyper_indices[1]])
    # phase all hyper spectra
    # dont norm them for T1 fitting
    hyper_spectra = []
    for spectrum in range(Nspec_hyper):
        spec = np.fft.fftshift(np.fft.fft(hyper_data.fids[spectrum, :]))
        hyper_spectra.append(np.real(spec * np.exp(1j * (final_phase_hyper * np.pi) / 180.0)))
    hyper_spectra = np.array(hyper_spectra)

    # integrate a selected peak
    if select_peak_ppm_hyper:
        center_ppm_hyper = select_peak_ppm_hyper
    else:
        # otherwise find largest peak
        center_hyper = np.squeeze(np.where(hyper_normed - np.max(hyper_normed) == 0))
        center_ppm_hyper = ppm_axis_hyper[center_hyper]

    # find integration bounds from input of integration width
    lower_bound_integration_ppm_hyper = np.abs(ppm_axis_hyper - (center_ppm_hyper - integration_width_hyper))
    lower_bound_integration_index_hyper = np.argmin(
        lower_bound_integration_ppm_hyper - np.min(lower_bound_integration_ppm_hyper))
    upper_bound_integration_ppm_hyper = np.abs(ppm_axis_hyper - (center_ppm_hyper + integration_width_hyper))
    upper_bound_integration_index_hyper = np.argmin(
        upper_bound_integration_ppm_hyper - np.min(upper_bound_integration_ppm_hyper))
    integrated_peak_roi_hyper = [lower_bound_integration_index_hyper, upper_bound_integration_index_hyper]
    # sorted so that lower index is first
    integrated_peak_roi_hyper.sort()

    # calculate the SNR of the first useable hyper spectrum, per default this is the first one measured
    # this SNR will be compared to the thermal SNR later to determine the polarization level
    SNR_hyper = np.sum(hyper_normed[integrated_peak_roi_hyper[0]:integrated_peak_roi_hyper[1]])

    # Now we also need the exponential decay constant to backcalculate the SNR over time (optional)

    # integrate the spectra, not the normalized spectra for the T1 fit, as otherwhise there could be fit issues due to high noise

    Hyper_Signal_for_T1_fit = np.array([np.sum(hyper_spectra[spectrum,integrated_peak_roi_hyper[0]:integrated_peak_roi_hyper[1]]) for spectrum in range(first_spec,Nspec_hyper)])
    # define the time axis during which the scans took place
    hyp_time_axis = np.arange(first_spec*TR_hyper, TR_hyper * Nspec_hyper, TR_hyper)
    # Fit function
    def exponential(x, M, T1):
        return M * np.exp(-x / T1)
    coeff, err = curve_fit(exponential, hyp_time_axis, Hyper_Signal_for_T1_fit, p0=(np.max(Hyper_Signal_for_T1_fit), 60))

    # flipangle correct this
    flipangle_corr_T1 = flipangle_corr(coeff[1], FA_hyper, TR_hyper)
    error_t1 = np.sqrt(np.diag(err))[1]

    # backcalculate the SNR from the first spectrum that was used using the decay constant of magnetization
    # Possible error case here: SNR could not decay with the same constant as the signal (differences in noise)
    # As this backcalculation is to be seen with caution it does not matter much here
    # However has to be kept in mind
    if T1_for_backcalculation:
        SNR_hyper_backcalculated = exponential(-time_to_diss, SNR_hyper, T1_for_backcalculation)
    else:
        SNR_hyper_backcalculated = exponential(-time_to_diss, SNR_hyper, flipangle_corr_T1)

    # print('Hyper SNR at t = ',time_to_diss,' s -- ',np.round(SNR_hyper))
    '------------------------------------------------------------------------------------------------------------'
    '------------------------------------------------------------------------------------------------------------'
    '------------------------------------------------------------------------------------------------------------'

    # Step 3: calculate thermal SNR
    # If multiple thermal averages are present --> mean them
    # Norm meaned spectrum to background region
    # Integrate peak to get SNR value
    FA_thermal = thermal_data.flipangle * np.pi / 180.

    ppm_axis_thermal = thermal_data.ppm_axis
    lower_bound_bg_thermal = np.abs(ppm_axis_thermal - bg_region_thermal[0])
    lower_bound_bg_index_thermal = np.argmin(lower_bound_bg_thermal - np.min(lower_bound_bg_thermal))
    upper_bound_bg_thermal = np.abs(ppm_axis_thermal - bg_region_thermal[1])
    upper_bound_bg_index_thermal = np.argmin(upper_bound_bg_thermal - np.min(upper_bound_bg_thermal))
    bg_region_thermal_indices = [lower_bound_bg_index_thermal, upper_bound_bg_index_thermal]
    bg_region_thermal_indices.sort()

    if thermal_data.number_of_spectra > 1:
        # mean thermal spectra if we have multiple that need to be averaged by us
        # i.e. SpinSolve Expert measurements

        nr_acq_pts = len(thermal_data.fids[0])
        time_axis = np.linspace(0, thermal_data.TR, nr_acq_pts) / 1000
        # linebroadening
        lb_fac = np.exp(-2 * np.pi * linebroadening * time_axis)
        # number of thermal spectra
        Nspec_thermal = thermal_data.number_of_spectra
        # apply linebroadening in case its wanted through the averaging function
        therm_fid = np.mean(thermal_data.fids,axis=0)
        # phase it
        therm_spec_complex = np.fft.fftshift(np.fft.fft(therm_fid*lb_fac))
        therm_spec_complex_baseline_corr = therm_spec_complex-np.mean(therm_spec_complex[bg_region_thermal_indices[0]:bg_region_thermal_indices[1]])
        if thermal_phase_input is False:
            Integrals_therm = []
            phases = np.linspace(0, 360, 1000)
            for phase in phases:
                itgl = np.max(np.real(therm_spec_complex_baseline_corr * np.exp(1j * (phase * np.pi) / 180.0)))
                Integrals_therm.append(itgl)

            final_phase_therm = phases[np.argmin(np.abs(Integrals_therm - np.max(Integrals_therm)))]
        else:
            final_phase_therm = thermal_phase_input
        phased_therm_spec = np.real(therm_spec_complex_baseline_corr * np.exp(1j * (final_phase_therm * np.pi) / 180.0))
        thermal_normed = (phased_therm_spec - np.mean(
            phased_therm_spec[bg_region_thermal_indices[0]:bg_region_thermal_indices[1]])) / np.std(
            phased_therm_spec[bg_region_thermal_indices[0]:bg_region_thermal_indices[1]])


    else:
        # in case we use Spinsolve-normal experiments or single shot thermal references
        Nspec_thermal = thermal_data.averages

        nr_acq_pts = len(thermal_data.fids)
        time_axis = np.linspace(0, thermal_data.TR, nr_acq_pts) / 1000
        # linebroadening
        lb_fac = np.exp(-2 * np.pi * linebroadening * time_axis)
        therm_spec_complex = np.fft.fftshift(np.fft.fft(thermal_data.fids*lb_fac))
        therm_spec_complex_baseline_corr = therm_spec_complex-np.mean(therm_spec_complex[bg_region_thermal_indices[0]:bg_region_thermal_indices[1]])
        # phase it
        if thermal_phase_input is False:
            Integrals_therm = []
            phases = np.linspace(0, 360, 1000)
            for phase in phases:
                itgl = np.max(np.real(therm_spec_complex_baseline_corr * np.exp(1j * (phase * np.pi) / 180.0)))
                Integrals_therm.append(itgl)

            final_phase_therm = phases[np.argmin(np.abs(Integrals_therm - np.max(Integrals_therm)))]
        else:
            final_phase_therm = thermal_phase_input
        phased_therm_spec = np.real(therm_spec_complex_baseline_corr * np.exp(1j * (final_phase_therm * np.pi) / 180.0))

        thermal_normed = (phased_therm_spec - np.mean(
            phased_therm_spec[bg_region_thermal_indices[0]:bg_region_thermal_indices[1]])) / np.std(
            phased_therm_spec[bg_region_thermal_indices[0]:bg_region_thermal_indices[1]])


    # in case we want to integrate a specific peak
    if select_peak_ppm_thermal:
        center_ppm_thermal = select_peak_ppm_thermal
    else:
        # find largest peak
        center_thermal = np.squeeze(np.where(thermal_normed - np.max(thermal_normed) == 0))
        center_ppm_thermal = ppm_axis_thermal[center_thermal]



    #  integrate around peak
    lower_bound_integration_ppm_thermal = np.abs(ppm_axis_thermal - (center_ppm_thermal - integration_width_thermal))
    lower_bound_integration_index_thermal = np.argmin(
        lower_bound_integration_ppm_thermal - np.min(lower_bound_integration_ppm_thermal))
    upper_bound_integration_ppm_thermal = np.abs(ppm_axis_thermal - (center_ppm_thermal + integration_width_thermal))
    upper_bound_integration_index_thermal = np.argmin(
        upper_bound_integration_ppm_thermal - np.min(upper_bound_integration_ppm_thermal))
    # from this we calculate the integrated peak region
    # sorted so that lower index is first
    integrated_peak_roi_thermal = [lower_bound_integration_index_thermal, upper_bound_integration_index_thermal]
    integrated_peak_roi_thermal.sort()


    SNR_thermal = np.sum(thermal_normed[integrated_peak_roi_thermal[0]:integrated_peak_roi_thermal[1]])

    '------------------------------------------------------------------------------------------------------------'
    '------------------------------------------------------------------------------------------------------------'
    '------------------------------------------------------------------------------------------------------------'
    # Step 4 : correction factors
    # correct for different flipangles between hyper and thermal measurement
    # correct for differences in molar concentration if thermal reference (i.e. Urea)
    # was used
    # correct for
    # dont have to correct for Receiver gain on the 1T tabletop NMR, cause we are looking at SNR and
    # by norming to the background noise level we are already pricing in the change in receiver gain (linear)
    # we did experiments where the receiver gain was changed and spectra of a 8M Urea phantom were acquired
    # after norming to background noise level the spectra have the same SNR, although the RX gain is different
    Receiver_Gain_thermal = thermal_data.rxgain
    Receicer_Gain_hyper = hyper_data.rxgain

    correction_factor = np.sqrt(Nspec_thermal) * (np.sin(FA_thermal) / np.sin(FA_hyper)) * (molarity_thermal / molarity_hyper)
    enhancement_factor = (SNR_hyper_backcalculated / SNR_thermal) * correction_factor

    '------------------------------------------------------------------------------------------------------------'
    '------------------------------------------------------------------------------------------------------------'
    '------------------------------------------------------------------------------------------------------------'
    # Step 5: compare and plot results

    Polarization_level = Pol_lvl_thermal * enhancement_factor
    Polarization_level_at_first_spec = Pol_lvl_thermal * (SNR_hyper / SNR_thermal) * correction_factor
    Polarization_level = np.round(Polarization_level * 100, 1)
    Polarization_level_at_first_spec = np.round(Polarization_level_at_first_spec * 100, 1)


    if print_output is True:
        print('--------------------------------------------------------------')
        print('Corrected observed T1=',np.round(coeff[1]),' s, for a flipangle of ',
        FA_hyper*180/ np.pi, ' ° and a TR of ', TR_hyper,' s ')
        print('Resulting in T1_corr = ',np.round(flipangle_corr_T1,1),' s')
        print('Thermal measurement TR:',TR_thermal,' s' )
        print('Receiver Gain difference - Hyper RX Gain = ', Receicer_Gain_hyper, ' vs Thermal RX Gain = ',
              Receiver_Gain_thermal)
        print('Molarity  difference - Hyper Sample 13C Molarity = ', molarity_hyper,
              ' vs Thermal Sample 13C Molarity  = ',
              molarity_thermal)
        print('Number of spectra  difference - Hyper Scan 1 sample vs Thermal scan ',
              Nspec_thermal, ' sample')
        print('Flipangle difference correction - Hyper flip angle ', FA_hyper * 180 / np.pi,
              ' ° - vs Thermal flip angle ', FA_thermal * 180 / np.pi, ' °')
        print('Enhancement factor from thermal to hyper', "{:.1e}".format(enhancement_factor))
        if T1_for_backcalculation:
            print('Externally used T1 from other fit function = ', np.round(T1_for_backcalculation, 1))
        else:
            print('T1_hyper_corr = ', np.round(flipangle_corr_T1, 1), 'pm', np.round(error_t1, 1), ' s')

        print('SNR_thermal normed to Molarity and Number of spectra', np.round(SNR_thermal*(molarity_hyper/molarity_thermal)/np.sqrt(Nspec_thermal),1))
        print('SNR_thermal / correction factor = ', np.round(SNR_thermal/correction_factor,3))
        print('--------------------------------------------------------------')
        print('THERMAL Polarization = ', Pol_lvl_thermal)
        print('SNR_thermal = ', np.round(SNR_thermal, 1))
        print('N spec thermal ', Nspec_thermal)
        print('sqrt(N spec thermal) ', np.sqrt(Nspec_thermal))
        print('SNR_hyper_backcalculated = ', np.round(SNR_hyper_backcalculated, 1))
        print('SNR_hyper = ', np.round(SNR_hyper, 1))
        print('--------------------------------------------------------------')
        time_of_first_spec = TR_hyper*first_spec
        print('HYPER - Polarization level of first spec at T = ', time_of_first_spec, ' s, is ', Polarization_level_at_first_spec, ' %')
        print('HYPER - Polarization level at T = ', -time_to_diss, ' s, is ', Polarization_level, ' %')

        print('--------------------------------------------------------------')
    else:
        pass

    if plot is True:

        fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(10, 4))
        ax[0].plot(thermal_data.ppm_axis, thermal_normed)
        ax[0].set_xlabel('ppm')
        ax[0].set_title('Thermal spectrum - ' + str(Nspec_thermal) + ' averages')
        ax[0].set_ylabel('SNR')
        ax[0].fill_between(
            [thermal_data.ppm_axis[bg_region_thermal_indices[0]], thermal_data.ppm_axis[bg_region_thermal_indices[1]]],
            np.min(thermal_normed), np.max(thermal_normed), alpha=0.3, color='C2', label='Background')
        ax[0].fill_between(
            [thermal_data.ppm_axis[integrated_peak_roi_thermal[0]],
             thermal_data.ppm_axis[integrated_peak_roi_thermal[1]]],
            np.min(thermal_normed), np.max(thermal_normed), alpha=0.3, color='C1', label='Peak integration')

        ax[1].plot(hyper_data.ppm_axis, hyper_normed)
        ax[1].set_xlabel('ppm')
        ax[1].set_title('First hyper spectrum')
        ax[1].set_ylabel('SNR')
        ax[1].fill_between(
            [hyper_data.ppm_axis[bg_region_hyper_indices[0]], hyper_data.ppm_axis[bg_region_hyper_indices[1]]],
            np.min(hyper_normed),
            np.max(hyper_normed), alpha=0.3, color='C2', label='Background')
        ax[1].fill_between(
            [hyper_data.ppm_axis[integrated_peak_roi_hyper[0]], hyper_data.ppm_axis[integrated_peak_roi_hyper[1]]],
            np.min(hyper_normed), np.max(hyper_normed), alpha=0.3, color='C1', label='Peak integration')
        ax[1].legend()

        ax[2].scatter(hyp_time_axis, Hyper_Signal_for_T1_fit, label='Data points')
        time_ax = np.arange(-time_to_diss, TR_hyper * Nspec_hyper, TR_hyper)
        ax[2].plot(time_ax, exponential(np.arange(-time_to_diss, TR_hyper * Nspec_hyper, TR_hyper), coeff[0], coeff[1]),
                   label='Fit - T1=' + str(np.round(coeff[1], 1)) + 's')
        if T1_for_backcalculation:
            ax[2].plot(time_ax,
                       exponential(np.arange(-time_to_diss, TR_hyper * Nspec_hyper, TR_hyper), coeff[0], T1_for_backcalculation),
                       label='T1 manual input =' + str(np.round(T1_for_backcalculation, 1)) + 's')
        else:
            ax[2].plot(time_ax,
                       exponential(np.arange(-time_to_diss, TR_hyper * Nspec_hyper, TR_hyper), coeff[0], flipangle_corr_T1),
                       label='T1 corrected =' + str(np.round(flipangle_corr_T1, 1)) + 's')


        ax[2].legend()
        ax[2].set_ylabel('Hyper Signal [a.u.]')
        ax[2].set_xlabel('Time since start of experiment [s]')
        ax[2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # second plot showing background levels

        fig2, ax2 = plt.subplots(2, 2, figsize=(10, 4), tight_layout=True)
        ax2[0, 0].plot(ppm_axis_thermal[bg_region_thermal_indices[0]:bg_region_thermal_indices[1]],
                       thermal_normed[bg_region_thermal_indices[0]:bg_region_thermal_indices[1]])
        ax2[0, 1].plot(ppm_axis_hyper[bg_region_hyper_indices[0]:bg_region_hyper_indices[1]],
                       hyper_normed[bg_region_hyper_indices[0]:bg_region_hyper_indices[1]])

        ax2[0, 0].set_title('Thermal BG region')
        ax2[0, 1].set_title('Hyper BG region')
        x_data, y_data, bin_size = Get_Hist(thermal_normed[bg_region_thermal_indices[0]:bg_region_thermal_indices[1]],
                                            25)
        ax2[1, 0].bar(x_data, y_data, width=bin_size)
        x_data, y_data, bin_size = Get_Hist(hyper_normed[bg_region_hyper_indices[0]:bg_region_hyper_indices[1]], 25)
        ax2[1, 1].bar(x_data, y_data, width=bin_size)
        ax2[1, 0].set_title('Thermal BG - Histogram')
        ax2[1, 1].set_title('Hyper BG - Histogram')
        ax2[1, 0].set_xlabel('SNR val')
        ax2[1, 1].set_xlabel('SNR val')
        ax2[0, 0].set_xlabel('ppm')
        ax2[0, 1].set_xlabel('ppm')
        ax2[1, 0].set_ylabel('Nr Points')
        ax2[1, 1].set_ylabel('Nr Points')

        # third plot showing peak integration regions

        fig2, ax2 = plt.subplots(2, 2, figsize=(10, 4), tight_layout=True)
        ax2[0, 0].plot(ppm_axis_thermal[integrated_peak_roi_thermal[0]:integrated_peak_roi_thermal[1]],
                       thermal_normed[integrated_peak_roi_thermal[0]:integrated_peak_roi_thermal[1]])
        ax2[0, 1].plot(ppm_axis_hyper[integrated_peak_roi_hyper[0]:integrated_peak_roi_hyper[1]],
                       hyper_normed[integrated_peak_roi_hyper[0]:integrated_peak_roi_hyper[1]])

        ax2[0, 0].set_title('Thermal Signal region')
        ax2[0, 1].set_title('Hyper Signal region')
        x_data, y_data, bin_size = Get_Hist(
            thermal_normed[integrated_peak_roi_thermal[0]:integrated_peak_roi_thermal[1]], 100)
        ax2[1, 0].bar(x_data, y_data, width=bin_size)
        x_data, y_data, bin_size = Get_Hist(hyper_normed[integrated_peak_roi_hyper[0]:integrated_peak_roi_hyper[1]],
                                            100)
        ax2[1, 1].bar(x_data, y_data, width=bin_size)
        ax2[1, 0].set_title('Thermal Signal - Histogram')
        ax2[1, 1].set_title('Hyper Signal - Histogram')
        ax2[1, 0].set_xlabel('SNR val')
        ax2[1, 1].set_xlabel('SNR val')
        ax2[0, 0].set_xlabel('ppm')
        ax2[0, 1].set_xlabel('ppm')
        ax2[1, 0].set_ylabel('Nr Points')
        ax2[1, 1].set_ylabel('Nr Points')



    else:
        pass

    return Polarization_level,Polarization_level_at_first_spec, SNR_thermal, SNR_hyper_backcalculated, Pol_lvl_thermal, enhancement_factor, flipangle_corr_T1



