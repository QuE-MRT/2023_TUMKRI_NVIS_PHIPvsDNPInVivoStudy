# Author: Luca Nagel, luca.nagel@tum.de
#         Wolfgang Gottwald, wolfgang.gottwald@tum.de

# Institution: Technical University of Munich
# Date of last Edit (dd.mm.yyyy): 24.05.2023


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from ..brukerexp import BrukerExp


class CPMG(BrukerExp):
    def __init__(self, path_or_BrukerExp):
        """Accepts directory path or BrukerExp object as input."""
        if isinstance(path_or_BrukerExp, BrukerExp):
            path_or_BrukerExp = path_or_BrukerExp.path

        super().__init__(path_or_BrukerExp)

    def calculate_T2_thermal(self):
        """
        Fits exponential to CPMG measurement to retrieve T2 value for a thermal T2 measurement which has lots
        of repetitions.

        Returns
        -------
        T2 : float
            T2 decay constant from exponential fit
        T2_err : float
            Error of T2 decay constant from exponential fit
        """

        echotrain = []
        integral = []
        n_echos = self.method["NEchoes"]
        # apparently it always acquires 128 points
        # if the spec matrix is smaller than 128
        spec_acq_pts = self.method["PVM_SpecMatrix"]
        if spec_acq_pts < 128:
            spec_acq_pts = 128
        else:
            pass
        echo_spacing = self.method["EchoSpacing"]
        n_reps = self.method["PVM_NRepetitions"]
        # average over repetitions
        fid_file = self.fid
        if n_reps > 1:
            avg_fid = []
            for n in range(0, n_reps, 1):
                avg_fid.append(
                    fid_file[
                        n * spec_acq_pts * n_echos : n * spec_acq_pts * n_echos
                        + spec_acq_pts * n_echos
                    ]
                )
            avg_fid = np.array(avg_fid)
            # mean the fid
            fid_file = np.mean(avg_fid, axis=0)
        else:
            pass

        # extract data
        for n in range(0, n_echos, 1):
            echotrain.append(
                fid_file[n * spec_acq_pts : n * spec_acq_pts + spec_acq_pts]
            )
            integral.append(
                np.abs(
                    np.sum(fid_file[n * spec_acq_pts : n * spec_acq_pts + spec_acq_pts])
                )
            )
        # form x axis for fit
        x_ax = np.arange(0, echo_spacing * n_echos, echo_spacing)
        # interpolate
        x_ax_itp = np.linspace(0, echo_spacing * n_echos, 10000)

        def exp(x, a, b, c):
            return a * np.exp(-x / b) + c

        # norm for easier fitting
        integral = integral / np.max(integral)
        coeff, err = curve_fit(exp, x_ax, integral, p0=(1, 20, 0.2))
        error = np.sqrt(np.diag(err))

        fig, ax = plt.subplots(1)
        ax.set_title(
            "CPMG - Echo spacing "
            + str(echo_spacing)
            + " , Echoes "
            + str(n_echos)
            + " , Repetitions "
            + str(n_reps)
        )
        ax.scatter(x_ax, integral, label="Data")
        ax.plot(
            x_ax_itp,
            exp(x_ax_itp, coeff[0], coeff[1], coeff[2]),
            label="Fit - T2 =" + str(np.round(coeff[1], 1)) + " ms",
        )
        ax.legend()
        ax.set_xlabel("[ms]")
        ax.set_ylabel("I [a.u.]")
        print(
            "T2=", np.round(coeff[1], 1), " plus minus ", np.round(error[1], 1), " ms"
        )
        T2 = coeff[1]
        T2_err = error[1]
        return T2, T2_err

    def calculate_T2_hyper(
        self, Sample_ID, starting_index=0, guessedT2=50000, ax_to_plot_to = False,savepath=None
    ):
        """
        Loads CPMG measurement and fits exponential to it to retrieve T2 value.

        Parameters
        -------
        Sample_ID : int or str
            Name of sample
        starting_index: int
            Where the fitting starts, maybe you want to leave out a few first data points
        guessedT2: int/float
            Estimate for the T2 in ms.
        ax_to_plot_to: bool
        savepath : bool / str, default None.
            path to which figure will be saved
        Returns
        -------
        T2 : float
            T2 decay constant from exponential fit
        T2_err : float
            Error of T2 decay constant from exponential fit
        """

        echotrain = []
        integral = []
        n_echos = self.method["NEchoes"]
        # apparently it always acquires 128 points
        # if the spec matrix is smaller than 128
        spec_acq_pts = self.method["PVM_SpecMatrix"]
        if spec_acq_pts < 128:
            spec_acq_pts = 128
        else:
            pass
        echo_spacing = self.method["EchoSpacing"]
        n_reps = self.method["PVM_NRepetitions"]
        # average over repetitions
        fid_file = self.fid
        if n_reps > 1:
            # only use first rep
            fid_file = fid_file[
                spec_acq_pts * n_echos : spec_acq_pts * n_echos + spec_acq_pts * n_echos
            ]
        else:
            pass

        # extract data
        for n in range(0, n_echos, 1):
            echotrain.append(
                fid_file[n * spec_acq_pts : n * spec_acq_pts + spec_acq_pts]
            )
            integral.append(
                np.abs(
                    np.sum(fid_file[n * spec_acq_pts : n * spec_acq_pts + spec_acq_pts])
                )
            )
        # form x axis for fit
        x_ax = np.arange(0, echo_spacing * n_echos, echo_spacing)
        # interpolate
        x_ax_itp = np.linspace(0, echo_spacing * n_echos, 10000)

        def exp(x, a, b, c):
            return a * np.exp(-x / b) + c

        # norm for easier fitting
        integral = integral / np.max(integral)
        # option to leave out first few points for the fit
        integral = integral[starting_index:]
        x_ax = x_ax[starting_index:]

        coeff, err = curve_fit(exp, x_ax, integral, p0=(1, guessedT2, 0.2))
        error = np.sqrt(np.diag(err))
        if ax_to_plot_to:
            ax = ax_to_plot_to
            ax.set_title(
                "Echo spacing " + str(echo_spacing) + " , Echoes " + str(n_echos)
            )
        else:
            fig, ax = plt.subplots(1)
            ax.set_title(
                "CPMG - Echo spacing " + str(echo_spacing) + " , Echoes " + str(n_echos)
            )
        ax.scatter(x_ax/1000, integral, label="Data", color="r")
        ax.plot(
            x_ax/1000,
            exp(x_ax/1000, coeff[0], coeff[1]/1000, coeff[2]),
            label="Fit - T2 ="
            + str(np.round(coeff[1]/1000, 1))
            + r"$\pm$"
            + str(np.round(error[1]/1000, 1))
            + " s",
        )
        ax.legend()
        ax.set_xlabel(" Time since start of experiment [s]")
        ax.set_ylabel("I [a.u.]")
        if savepath:
            plt.savefig(
                savepath + "Sample_" + str(Sample_ID) + "_T2_measurement_7T.png"
            )
        print(
            "7T measurement -- T2= ",
            np.round(coeff[1] / 1000, 1),
            " plus minus ",
            np.round(error[1] / 1000, 1),
            " s",
        )
        T2 = coeff[1]
        T2_err = error[1]

        return T2, T2_err
