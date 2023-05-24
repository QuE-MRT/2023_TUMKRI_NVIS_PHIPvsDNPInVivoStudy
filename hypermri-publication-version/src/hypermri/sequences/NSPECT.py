# Authors: Luca Nagel, luca.nagel@tum.de
#          Wolfgang Gottwald, wolfgang.gottwald@tum.de

# Institution: Technical University of Munich
# Date of last Edit (dd.mm.yyyy): 24.05.2023

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import ipywidgets as widgets

from ..brukerexp import BrukerExp


class NSPECT(BrukerExp):
    def __init__(self, path_or_BrukerExp):
        """Accepts directory path or BrukerExp object as input."""
        if isinstance(path_or_BrukerExp, BrukerExp):
            path_or_BrukerExp = path_or_BrukerExp.path

        super().__init__(path_or_BrukerExp)

        # in case this is dual channel data
        if self.n_receivers == 2:
            self.fid_ch1 = self.get_two_channel_fids()[0]
            self.fid_ch2 = self.get_two_channel_fids()[1]
            self.correct_phase = self.find_phase_shift_dual_channel(False)
            self.phased_fid = self.apply_phase_shift_dual_channel(self.correct_phase)

            (
                self.ppm_axis,
                self.spec_phased,
                self.spec_ch1,
                self.spec_ch2,
            ) = self.get_spec_non_localized_spectroscopy_dual_channel(
                self.correct_phase
            )
        else:
            # its single channel data
            (
                self.ppm_axis,
                self.spec,
                self.fids,
                self.complex_spec,
            ) = self.get_spec_non_localized_spectroscopy()

    # non dual channel functions
    # TODO make a function similar to get_two_channel_fids but that just gets the one channel fid
    def get_spec_non_localized_spectroscopy(self, LB=0, cut_off=70):
        """
        Calculates spectra and ppm axis for non localized spectroscopy measurements with repetitions.
        Parameters
        ----------
        LB : float, optional
            linebroadening applied in Hz, default is 0
        cut_off : float, optional
            number of first points the fid is cut off as these are only noise, default is 70

        Returns
        -------
        ppm : array
            ppm-axis calculated from meta data
        spec : array
            magnitude spectra in a n-dimensional array
        fids: array
            linebroadened fids in a n-dimensional array
        complex_spec: array
            complex spectra in n-dimensional array
        """

        # if we dont have rawdatajob0 file we load fid
        if len(self.rawdatajob0) == 0:
            fid = self.fid
        else:
            fid = self.rawdatajob0
        center = float(self.method["PVM_FrqWorkPpm"][0])
        bw = self.method["PVM_SpecSW"]
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]
        NR = self.method["PVM_NRepetitions"]
        time_ax = np.linspace(0, ac_time, ac_points - cut_off) / 1000
        sigma = 2.0 * np.pi * LB

        ppm = np.linspace(
            center - bw / 2,
            center + bw / 2,
            ac_points - cut_off,
        )
        fids = np.zeros((NR, ac_points - cut_off), dtype=complex)
        spec = np.zeros((NR, ac_points - cut_off))
        complex_spec = np.zeros((NR, ac_points - cut_off), dtype=complex)

        rep_counter = 0
        while rep_counter < NR:
            test_spec = np.fft.fftshift(
                np.fft.fft(
                    fid[
                        cut_off
                        + rep_counter * ac_points : ac_points
                        + rep_counter * ac_points
                    ]
                    * np.exp(-sigma * time_ax)
                )
            )
            spec[rep_counter, :] = np.abs(test_spec)
            complex_spec[rep_counter, :] = test_spec
            fids[rep_counter, :] = fid[
                cut_off + rep_counter * ac_points : ac_points + rep_counter * ac_points
            ] * np.exp(-sigma * time_ax)
            rep_counter += 1

        return ppm, spec, fids, complex_spec

    def phase_correct_fid(self, number=0):
        """
        Interactively allows phasing of fids.
        Parameters
        ----------
        number: int, number of the fid that should be phased, in case there are repetitons.
        """

        fid = self.fids[number, :]
        spec_complex = np.fft.fftshift(np.fft.fft(fid))
        # perform a baseline correction
        spec_base_line_corr = spec_complex - np.mean(spec_complex)
        # phase the real spectrum
        Integrals_th = []
        phases = np.linspace(0, 360, 50)
        for phase in phases:
            itgl = np.sum(
                np.real(spec_base_line_corr * np.exp(1j * (phase * np.pi) / 180.0))
            )
            Integrals_th.append(itgl)
        initial_guess_phase = phases[
            np.argmin(np.abs(Integrals_th - np.max(Integrals_th)))
        ]

        fig, ax = plt.subplots(1, figsize=(12, 4), tight_layout=True)

        (line_real,) = ax.plot(
            np.real(
                spec_base_line_corr * np.exp(1j * (initial_guess_phase * np.pi) / 180.0)
            ),
            label="Real-phased",
            color="k",
        )
        (line_abs,) = ax.plot(
            np.abs(spec_base_line_corr) - np.mean(np.abs(spec_base_line_corr)),
            label="Magnitude",
            color="r",
        )
        ax.hlines(
            0,
            0,
            len(spec_base_line_corr),
            linestyles="dashed",
            alpha=0.3,
            color="b",
            label="Baseline",
        )
        # ax.fill_between([8000, 9000],np.min(np.real(spec_base_line_corr)),np.max(np.real(spec_base_line_corr)), alpha=0.3, color='C2', label='Background')
        ax.set_xlabel("Points")
        ax.set_ylabel("I [a.u.]")
        ax.set_title("Phased with " + str(np.round(phase, 1)) + " °")
        ax.legend()

        @widgets.interact(phase=(0, 360, 0.1))
        def update(phase):
            line_real.set_ydata(
                np.real(spec_base_line_corr * np.exp(1j * (phase * np.pi) / 180.0))
            )
            line_abs.set_ydata(
                np.abs(spec_base_line_corr) - np.mean(np.abs(spec_base_line_corr))
            )
            ax.set_title("Phased with " + str(np.round(phase, 1)) + " °")

    # Analysis functions
    def calculate_T1(
        self,
        first_spec,
        last_spec,
        Sample_ID,
        savepath=None,
        peak_ppm=False,
        integration_width=50,
        ax_to_plot_to = False
    ):

        ppm_axis_hyper, spec, fids,complex_spec = self.get_spec_non_localized_spectroscopy(0, 70)
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]

        FA = float(self.method["ExcPulse1"][2])
        TR = float(self.method["PVM_RepetitionTime"]) / 1000

        time_ax = np.arange(TR * first_spec, last_spec * TR, TR)

        if peak_ppm:
            center_hyper = np.squeeze(np.where(spec - peak_ppm == 0))[1]
            center_ppm_hyper = peak_ppm
        else:
            center_hyper = np.squeeze(np.where(spec - np.max(spec) == 0))[1]

            center_ppm_hyper = ppm_axis_hyper[center_hyper]

        lower_bound_integration_ppm_hyper = np.abs(
            ppm_axis_hyper - (center_ppm_hyper - integration_width)
        )
        lower_bound_integration_index_hyper = np.argmin(
            lower_bound_integration_ppm_hyper
            - np.min(lower_bound_integration_ppm_hyper)
        )
        upper_bound_integration_ppm_hyper = np.abs(
            ppm_axis_hyper - (center_ppm_hyper + integration_width)
        )
        upper_bound_integration_index_hyper = np.argmin(
            upper_bound_integration_ppm_hyper
            - np.min(upper_bound_integration_ppm_hyper)
        )
        # from this we calculate the integrated peak region
        # sorted so that lower index is first
        integrated_peak_roi_hyper = [
            lower_bound_integration_index_hyper,
            upper_bound_integration_index_hyper,
        ]
        integrated_peak_roi_hyper.sort()
        SNR = []
        for n in range(first_spec, last_spec, 1):
            SNR.append(
                np.sum(
                    spec[n, integrated_peak_roi_hyper[0] : integrated_peak_roi_hyper[1]]
                )
            )
        # norm to one for T1
        # SNR=SNR/SNR[0]
        def exp(x, a, T1):
            return a * np.exp(-x / T1)

        coeff, err = curve_fit(exp, time_ax, SNR, p0=(np.max(SNR), 70))
        T1error = np.sqrt(np.diag(err))[1]

        T1 = 1 / ((1 / coeff[1]) + (np.log(np.cos(FA * np.pi / 180)) / TR))
        print(
            "7T measurement -- T1= ",
            np.round(T1, 1),
            "plus minus ",
            np.round(T1error, 1),
            " s",
        )
        if ax_to_plot_to:
            ax = ax_to_plot_to
        else:

            fig, ax = plt.subplots(1)
        ax.scatter(time_ax, SNR, label="Data", color="r")
        ax.plot(
            time_ax,
            exp(time_ax, coeff[0], coeff[1]),
            label="Fit, T1 = "
            + str(np.round(coeff[1], 1))
            + r"$\pm$"
            + str(np.round(T1error, 1))
            + " s",
        )
        ax.set_xlabel("Time since start of experiment [s] ")
        ax.set_ylabel("I [a.u.]")
        ax.set_title(
            "FA ="
            + str(FA)
            + "°, TR ="
            + str(TR)
            + " s, NR = "
            + str(last_spec - first_spec)
        )

        ax.legend()

        if savepath:
            plt.savefig(
                savepath + "Sample_" + str(Sample_ID) + "_t1_measurement_7T.png"
            )
        return T1, T1error

    # dual channel functions
    def get_two_channel_fids(self, cut_off=70):
        """
        Splits recorded data into two channels for measurements with a 2-channel receiver coil.
        Returns
        -------
        fid_ch1: np.array, shape = (NR,ac_points)
        fid_ch2: np.array, shape = (NR,ac_points)
        """

        if len(self.rawdatajob0) > 0:
            data = self.rawdatajob0
        else:
            data = self.fid

        ac_points = self.method["PVM_SpecMatrix"]
        NR = self.method["PVM_NRepetitions"]
        fid_ch1 = np.zeros((NR, ac_points - cut_off), dtype=complex)
        fid_ch2 = np.zeros((NR, ac_points - cut_off), dtype=complex)
        for i in np.arange(0, NR * 2, 2):
            nn = int(i / 2)
            fid_ch1[nn, :] = data[i * ac_points + cut_off : i * ac_points + ac_points]
            # counter to accurately put data at right points, because we can not just divide
            # by 2 like we did for channel 1
        count = 0
        for i in np.arange(1, NR * 2, 2):
            fid_ch2[count, :] = data[
                i * ac_points + cut_off : i * ac_points + ac_points
            ]
            count += 1

        return fid_ch1, fid_ch2

    def find_phase_shift_dual_channel(self, LB=0, cut_off=70, plot=False):
        """
        Finds the phase shift between data from a dual channel coil

        Parameters
        -------
        lb: float, optional
            Linebroadening applied to spectra in Hz.
        cut_off: int, optional
            Number of points that are left out for recorded fid as they are just noise at the beginning, default is 70.
        plot: bool, optional
            if a plot of the result is wanted for QA, can be turned to True

        Returns
        -------
        final_phase: float, phase in degree that maximizes integral of both channels
        """

        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]
        time_ax = np.linspace(0, ac_time, ac_points - cut_off) / 1000
        fid_ch1, fid_ch2 = self.get_two_channel_fids(cut_off)

        # finding out which of the two channels has the highest signal at which repetition
        # so that we phase the repetition that has the most signal
        # first integrate each repetition for each of the two channels seperately
        integral_ch1 = []
        integral_ch2 = []
        for n in range(self.method["PVM_NRepetitions"]):
            integral_ch1.append(np.sum(np.abs(fid_ch1[n, :])))
            integral_ch2.append(np.sum(np.abs(fid_ch2[n, :])))
        # find out at which repetition the signal is maximized
        max_signal_rep_ch1 = np.where(np.abs(integral_ch1 - np.max(integral_ch1)) == 0)[
            0
        ][0]
        max_signal_rep_ch2 = np.where(np.abs(integral_ch2 - np.max(integral_ch2)) == 0)[
            0
        ][0]
        # check which channel has the larger difference (i.e. more signal)
        # we cant just compare the max values cause the background offset is different
        # i.e. channel 2 could have the same absolute intensity maximum but relatively
        # it has lower signal
        signal_diff_ch1 = np.max(integral_ch1) - np.min(integral_ch1)
        signal_diff_ch2 = np.max(integral_ch2) - np.min(integral_ch2)
        # selecting which index to use
        if signal_diff_ch1 > signal_diff_ch2:
            max_signal_rep = max_signal_rep_ch1
        else:
            max_signal_rep = max_signal_rep_ch2

        ch_1 = fid_ch1[max_signal_rep, :]
        ch_2 = fid_ch2[max_signal_rep, :]
        # Can apply linebroadening here for nicer plots
        sigma = 2 * np.pi * LB
        ch_1_spec = np.abs(np.fft.fftshift(np.fft.fft(ch_1 * np.exp(-sigma * time_ax))))
        ch_2_spec = np.abs(np.fft.fftshift(np.fft.fft(ch_2 * np.exp(-sigma * time_ax))))
        # ppm scale
        center = float(self.method["PVM_FrqWorkPpm"][0])
        bw = self.method["PVM_SpecSW"]
        ppm = np.linspace(center - bw / 2, center + bw / 2, ac_points - cut_off)
        # finding optimal phase shift between channels
        Integrals = []
        phases = np.linspace(0, 360, 1000)
        for phase in phases:
            itgl = np.sum(np.abs(ch_1 * np.exp(1j * (phase * np.pi) / 180.0) + ch_2))
            Integrals.append(itgl)

        final_phase = phases[np.argmin(np.abs(Integrals - np.max(Integrals)))]
        # Optional plotting for QA
        if plot is True:
            fig, (ax, ax2) = plt.subplots(1, 2)
            ax.plot(phases, Integrals / np.max(Integrals))
            ax.set_xlabel(r"$\phi$ [rad]")
            ax.set_ylabel("Integral of ch_1 phaseshifted against ch_2")
            ax.vlines(
                final_phase,
                np.min(Integrals / np.max(Integrals)),
                1,
                color="orange",
            )
            ax.set_title(r"$\phi$ = " + str(np.round(final_phase, 1)) + "deg")

            # now plot spectra
            best_spec = np.abs(
                np.fft.fftshift(
                    np.fft.fft(
                        (ch_1 * np.exp(1j * (final_phase * np.pi) / 180.0) + ch_2)
                        * np.exp(-sigma * time_ax)
                    )
                )
            )

            ax2.plot(ppm, ch_1_spec / np.max(best_spec), label="Ch_1 spec")
            ax2.plot(ppm, ch_2_spec / np.max(best_spec), label="Ch_2 spec")
            ax2.plot(
                ppm,
                best_spec / np.max(best_spec),
                label="Both Channels spec",
            )
            ax2.set_xlabel(r"$\sigma$[ppm]")
            ax2.set_ylabel("I[a.u.]")
            ax2.legend(loc="best", ncol=1)
            ax2.set_title("Spectra from dual channel data")
            ax2.set_xlim([np.max(ppm), np.min(ppm)])
            minimum = np.argmin(np.abs(Integrals - np.min(Integrals)))
            fig.suptitle("NSPECT")
            plt.tight_layout()
        else:
            pass

        return final_phase

    def apply_phase_shift_dual_channel(self, phase_shift, cut_off=70):
        """
        Applies phase correction by given value.
        Parameters
        ----------
        phase_shift: float, in degree, phase by which channel 1 is shifted against channel 2
        LB: float, linebroadening applied to spectra
        cut_off: int, number of points at beginning of fid that are left out.
        Returns
        -------
        phased_fid
        """

        ac_points = self.method["PVM_SpecMatrix"]
        NR = self.method["PVM_NRepetitions"]

        ch_1 = self.fid_ch1
        ch_2 = self.fid_ch2
        phased_fid = np.zeros((NR, ac_points - cut_off), dtype=complex)
        for n in np.arange(0, NR):
            phased_fid[n, :] = (
                ch_1[n, :] * np.exp(1j * (phase_shift * np.pi) / 180.0) + ch_2[n, :]
            )
        return phased_fid

    # FIXME: if one chooses a cut_off that is not 70, the function does not work
    # FIXME:  because the fid file is always cut off to 70 --> get_two_channel_fids
    # FIXME: re write this a bit more smartly
    def get_spec_non_localized_spectroscopy_dual_channel(self, LB=0, cut_off=70):
        """
        Calculates spectra and ppm axis for non localized spectroscopy measurements with repetitions and dual
        channel data.
        Parameters
        ----------
        LB : float, optional
            linebroadening applied in Hz, default is 0
        cut_off : float, optional
            number of first points the fid is cut off as these are only noise, default is 70

        Returns
        -------
        ppm_axis : np.array
            ppm-scale for spectra
        spec_phased : np.array
            spectra for each repetition calculated from combining both channels with the
            optimal phase according to find_phase_shift_dual_channel
        spec_ch1 : np.array
            spectra for each repetition for data from channel 1
        spec_ch2 : np.array
            spectra for each repetition for data from channel 2
        """

        center = float(self.method["PVM_FrqWorkPpm"][0])
        bw = self.method["PVM_SpecSW"]
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]
        NR = self.method["PVM_NRepetitions"]
        time_ax = np.linspace(0, ac_time, ac_points - cut_off) / 1000
        sigma = 2.0 * np.pi * LB

        ppm_axis = np.linspace(center - bw / 2, center + bw / 2, ac_points - cut_off)

        spec_phased = np.zeros((NR, ac_points - cut_off))
        spec_ch1 = np.zeros((NR, ac_points - cut_off))
        spec_ch2 = np.zeros((NR, ac_points - cut_off))

        for rep_counter in range(0, NR, 1):
            spec_phased[rep_counter, :] = np.abs(
                np.fft.fftshift(
                    np.fft.fft(
                        self.phased_fid[rep_counter, :] * np.exp(-sigma * time_ax)
                    )
                )
            )
            spec_ch1[rep_counter, :] = np.abs(
                np.fft.fftshift(
                    np.fft.fft(self.fid_ch1[rep_counter, :] * np.exp(-sigma * time_ax))
                )
            )
            spec_ch2[rep_counter, :] = np.abs(
                np.fft.fftshift(
                    np.fft.fft(self.fid_ch2[rep_counter, :] * np.exp(-sigma * time_ax))
                )
            )

        return ppm_axis, spec_phased, spec_ch1, spec_ch2

    # analysis for dual channel
    def plot_spec_non_localized_spectroscopy_dual_channel(self, linebroadening=0):
        """
        Plots dual channel data for a NSPECT or Singlepulse sequence interactively
        Parameters
        ----------
        linebroadening: float, optional
            Linebroadening applied to spectra in Hz, default is 0.

        Returns
        -------
        """
        ac_time = self.method["PVM_SpecAcquisitionTime"]
        ac_points = self.method["PVM_SpecMatrix"]
        NR = self.method["PVM_NRepetitions"]
        time_ax = np.linspace(0, ac_time, ac_points - 70) / 1000
        sigma = 2.0 * np.pi * linebroadening

        (
            ppm_axis,
            spec_phased,
            spec_ch1,
            spec_ch2,
        ) = self.get_spec_non_localized_spectroscopy_dual_channel(linebroadening, 70)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        @widgets.interact(rep=(0, NR - 1, 1))
        def update(rep=0):
            [l.remove() for l in ax[0].lines]
            [l.remove() for l in ax[0].lines]
            [l.remove() for l in ax[0].lines]

            ax[0].plot(
                time_ax,
                np.real(self.fid_ch1[rep, :] * np.exp(-sigma * time_ax)),
                label="Re(Fid_Ch1)",
                color="r",
            )
            ax[0].plot(
                time_ax,
                np.real(self.fid_ch2[rep, :] * np.exp(-sigma * time_ax)),
                label="Re(Fid_Ch2)",
                color="b",
            )
            ax[0].plot(
                time_ax,
                np.real(self.phased_fid[rep, :] * np.exp(-sigma * time_ax)),
                label="Re(PhasedFid)",
                color="k",
            )
            ax[0].set_title("Fids")

            [l.remove() for l in ax[1].lines]
            [l.remove() for l in ax[1].lines]
            [l.remove() for l in ax[1].lines]

            ax[1].plot(ppm_axis, spec_ch1[rep, :], label="Ch1", color="r")
            ax[1].plot(ppm_axis, spec_ch2[rep, :], label="Ch2", color="b")
            ax[1].plot(
                ppm_axis, spec_phased[rep, :], label="Phased spectrum", color="k"
            )
            ax[1].set_title("Spectra")

        ax[0].legend()
        ax[1].legend()
        fig.suptitle("Linebroadening " + str(linebroadening) + " Hz")
