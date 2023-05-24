# Author:  Wolfgang Gottwald, wolfgang.gottwald@tum.de


# Institution: Technical University of Munich
# Date of last Edit (dd.mm.yyyy): 24.05.2023
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

from ..brukerexp import BrukerExp


class Singlepulse(BrukerExp):
    def __init__(self, path_or_BrukerExp):
        """Accepts directory path or BrukerExp object as input."""
        if isinstance(path_or_BrukerExp, BrukerExp):
            path_or_BrukerExp = path_or_BrukerExp.path

        super().__init__(path_or_BrukerExp)
        (
            self.ppm_axis,
            self.spec,
            self.fids,
            self.complex_spec,
        ) = self.get_spec_non_localized_spectroscopy()

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
        number: int, number of the fid that should be phase, in case there are repetitons.
        """

        fid = self.fids[number, :]
        spec_complex = np.fft.fftshift(np.fft.fft(fid))
        # perform a baseline correction
        # FIXME this has issues with solenoid measurements because there is no homogeneous baseline background
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
            np.mean(spec_complex),
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

        @widgets.interact(phase=(0, 360, 0.05))
        def update(phase):
            line_real.set_ydata(
                np.real(spec_base_line_corr * np.exp(1j * (phase * np.pi) / 180.0))
            )
            line_abs.set_ydata(
                np.abs(spec_base_line_corr) - np.mean(np.abs(spec_base_line_corr))
            )
            ax.set_title("Phased with " + str(np.round(phase, 1)) + " °")
