# Authors: Andre Wendlinger, andre.wendlinger@tum.de
#          Luca Nagel, luca.nagel@tum.de
#          Wolfgang Gottwald, wolfgang.gottwald@tum.de


# Institution: Technical University of Munich
# Date of last Edit (dd.mm.yyyy): 24.05.2023

"""
Base class used to read in one Bruker experiment/<expno> directory (see below).

Have a look at the ParaVision Manual for more information on the scan-data folder
structure.

Includes parts of functions copied and adapted with permission (https://github.com/jdoepfert/brukerMRI/issues/4)
from brukerMRI package (https://github.com/jdoepfert/brukerMRI),
namely: __ParseArray, __ParseSinglevalue, __ReadParamFile
We included the License file in this repository under ExternalLicenses/LICENSE_brukerMRI.
"""

import os
import re
import numpy as np
from pathlib import Path
from brukerapi.dataset import Dataset


class BrukerExp:
    """Base class for experiments conducted on Bruker preclinical systems.

    ParaVision dataset paths have the following form:

    <DataPath>/<name>/<expno>/pdata/<procno>

    <DataPath>
    This is an arbitrary path in the directory tree that is used as root for
    ParaVision datasets.
    <name>
    This directory contains all information about a study. The name is created by
    ParaVision. The maximum length of the directory name is 64 characters.
    <expno>
    The expno directory contains all information to describe one experiment. The
    directory name is the experiment number (usually an integer). The data from
    every new mri-sequence executed during the MR scan is stored in a new expno
    directory.
    <procno>
    The procno directory contains all information about a reconstructed or derived
    image series of the experiment. Several procnos may exist which contain
    different derived image series (e.g. different reconstructions, parameter
    images, etc.). The directory name is the processed images number.

    This class reads in a single <expno> directory.

    Attributes
    ----------
    TODO

    METHODS
    -------
    TODO
    """

    def __init__(self, exp_folder_path, load_data=True):
        """
        Initializer of the class.

        Parameters
        ----------
        exp_folder_path : str
            Complete path to measurement/experiment folder.

        Raises
        ------
        FileNotFoundError
            If path provided does not lead to a directory containing certain
            files identifying it as a bruker scan folder. One of the following
            key files must be included: "pdata", "fid", "acqp" or "seq2d".
        """
        if self.__is_valid_scan_dir(exp_folder_path):
            # store path
            self.path = Path(exp_folder_path)
            # load meta data files
            self.method = self.__ReadParamFile("method")
            self.acqp = self.__ReadParamFile("acqp")
            # get VisuExperimentNumber (E-number)
            self.ExpNum = self.__getVisuExperimentNumber()
            # define the measurement type
            self.type = self.__GetMeasurementType()
            # list of names of all recon folders inside 'pdata'
            self.recon_names = self.__FindRecons()
            if load_data:
                # load raw data into the instance
                self.fid = self.Load_fid_file()
                self.rawdatajob0 = self.Load_rawdatajob0_file()
                # load reconstructed data stored in '.../pdata/1'
                self.seq2d = self.Load_2dseq_file()
                # add number of receiver channels
                self.n_receivers = self.method["PVM_EncNReceivers"]
        else:
            raise FileNotFoundError(
                f"Folder {exp_folder_path} does not exist or is not a bruker experiment."
            )

    def __repr__(self):
        return f"<{__package__}.{self.__class__.__name__} name={self.acqp['ACQ_scan_name']}>"

    def __str__(self):
        return f"{self.acqp['ACQ_scan_name'][1:-1]}"  # drop encasing arrows <>

    @staticmethod
    def __is_valid_scan_dir(path):
        """Check if directory provided is a valid bruker scan directory."""
        if os.path.isdir(path):
            content = os.listdir(path)
            if len(set(content).intersection(["pdata", "fid", "acqp"])) > 0:
                return True
        return False

    @staticmethod
    def __is_valid_recon_dir(path):
        """Check if directory provided is a 'pdata' reconstruction directory."""
        if os.path.isdir(path):
            content = os.listdir(path)
            if len(set(content).intersection(["2dseq"])) > 0:
                return True
        return False

    def __getVisuExperimentNumber(self) -> int:
        """Try to get experiment number. Returns -1 when fails.

        First try to convert folder name into experiment number.

        In case that fails, falls back to extracting the number from the
        ACQ_scan_name (see self.__str__) using regular expressions.

        In case no experiment number is found inside the name, return -1.
        """
        # try to get ExpNum from scan folder name
        if self.path.name.isdigit():
            return int(self.path.name)

        # try to match last (E...) in scan name using regex
        match = re.match(r".*\(E(\d+)\)", str(self))

        return int(match.group(1)) if match else -1

        # # search for visu_pars file, if it exists extract VisuExperimentNumber:
        # # ISSUE: sometimes number is greater than 90000
        # visufname = os.path.join(self.path, "visu_pars")
        # if os.path.isfile(visufname):
        #     with open(visufname, "r") as f:
        #         for line in f:
        #             if line.startswith("##$VisuExperimentNumber"):
        #                 return int(line.split("=")[1].strip())

    def __ParseArray(self, current_file, line):
        """
        Parsing parameter files.

        Needed by ReadParamFile, copied from Bruker MRI package
        https://github.com/jdoepfert/brukerMRI
        """
        # extract the arraysize and convert it to numpy
        line = line[1:-2].replace(" ", "").split(",")
        arraysize = np.array([int(x) for x in line])
        # then extract the next line
        vallist = current_file.readline().split()
        # if the line was a string, then return it directly
        try:
            float(vallist[0])
        except ValueError:
            return " ".join(vallist)
        # include potentially multiple lines
        while len(vallist) != np.prod(arraysize):
            vallist = vallist + current_file.readline().split()
        # try converting to int, if error, then to float
        try:
            vallist = [int(x) for x in vallist]
        except ValueError:
            vallist = [float(x) for x in vallist]
        # convert to numpy array
        if len(vallist) > 1:
            return np.reshape(np.array(vallist), arraysize)
        # or to plain number
        else:
            return vallist[0]

    def __ParseSingleValue(self, val):
        """
        Parsing parameter files.
        Needed by ReadParamFile, copied from Bruker MRI package
        https://github.com/jdoepfert/brukerMRI
        """
        try:  # check if int
            result = int(val)
        except ValueError:
            try:  # then check if float
                result = float(val)
            except ValueError:
                # if not, should  be string. Remove  newline character.
                result = val.rstrip("\n")
        return result

    def __ReadParamFile(self, file):
        """
        Reads a Bruker MRI experiment's method or acqp file to a
        dictionary.
        Copied from Bruker MRI package
        https://github.com/jdoepfert/brukerMRI
        """
        try:
            param_dict = {}
            filepath = os.path.join(self.path, file)
            with open(filepath, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break

                    # when line contains parameter
                    if line.startswith("##$"):
                        (param_name, current_line) = line[3:].split("=")  # split at "="

                        # if current entry (current_line) is arraysize
                        if current_line[0:2] == "( " and current_line[-3:-1] == " )":
                            value = self.__ParseArray(f, current_line)

                        # if current entry (current_line) is struct/list
                        elif current_line[0] == "(" and current_line[-3:-1] != " )":
                            # if necessary read in multiple lines
                            while current_line[-2] != ")":
                                current_line = current_line[0:-1] + f.readline()

                            # parse the values to a list
                            value = [
                                self.__ParseSingleValue(x)
                                for x in current_line[1:-2].split(", ")
                            ]
                        # otherwise current entry must be single string or number
                        else:
                            value = self.__ParseSingleValue(current_line)

                        # save parsed value to dict
                        param_dict[param_name] = value
        except FileNotFoundError:
            print(f"No file available at {self.path}")
            param_dict = {}

        return param_dict

    def __GetMeasurementType(self):
        """Returns measurement type from methof file as string"""
        return self.method["Method"]

    def __FindRecons(self):
        """Find all recon-folder in pdata and return their names in a list."""
        pdata_path = os.path.join(self.path, "pdata")
        pdata_content = os.listdir(pdata_path)

        recon_folder_names = [
            recon_name
            for recon_name in pdata_content
            if self.__is_valid_recon_dir(os.path.join(pdata_path, recon_name))
        ]

        recon_folder_names.sort()

        return recon_folder_names

    def Load_fid_file(self):
        """Read fid file and return it as np.array.


        If multi slice data is loaded into topspin (i.e. SP with repetitions)
        the fid file is replaced by the ser file, which is loaded similarly.

        Therefore, in case there is no fid file but an ser file, try to load ser
        instead.

        Returns
        -------
        np.array
            complex fid file
        """
        fid_path = os.path.join(self.path, "fid")
        if os.path.exists(fid_path):
            with open(fid_path, "rb") as f:
                try:
                    raw_file = np.fromfile(f, dtype=np.int32, count=-1)
                except Exception as e:
                    # f"Error loading file {fid_path}: {e}"
                    return np.array([])
                else:
                    complex_file = raw_file[0::2] + 1j * raw_file[1::2]
            return complex_file

        ser_path = os.path.join(self.path, "ser")
        if os.path.exists(ser_path):
            with open(ser_path, "rb") as f:
                try:
                    raw_file = np.fromfile(f, dtype=np.int32)
                except Exception as e:
                    # f"Error loading file {ser_path}: {e}"
                    return np.array([])
                else:
                    complex_file = raw_file[0::2] + 1j * raw_file[1::2]
            return complex_file

        # "Error: FID and SER files not found in {self.path}"
        return np.array([])

    def Load_rawdatajob0_file(self):
        """
        Read the rawdata.job0 file and return it as a np.array.


        Returns
        -------
        complex_file: complex fid
        """
        try:
            with open(os.path.join(self.path, "rawdata.job0"), "r") as f:
                # The following part is for figuring out the issue with loading
                # job or fid raw data. It gives 1.28x too many points
                # blocksize/number of spectral points (not complex, CSI) = 1.28
                raw_file = np.fromfile(f, dtype=np.int32, count=-1)
                complex_file = raw_file[0::2] + 1j * raw_file[1::2]
        except FileNotFoundError:
            complex_file = np.array([])
            pass
        return complex_file

    def Load_2dseq_file(self, recon_num=1):
        """
        Load 2dseq file and return it as np.array.

        To load in a specific reconstruction use the argument recon_num.
        Note: This method makes use of the brukerapi package.

        Parameters
        ----------
        recon_num : int, optional
            number of reconstruction, by default 1.

        Returns
        -------
        out : np.array
            Complex array of the 2dseq file.
        """
        try:
            seq2d = Dataset(
                os.path.join(self.path, "pdata", str(recon_num), "2dseq")
            ).data

        except FileNotFoundError:
            seq2d = np.array([])
        return seq2d

    def open_method(self):
        import os, time, tempfile, pprint

        tmp = tempfile.NamedTemporaryFile(delete=False)

        try:
            print(f"Dumping tmp here:", tmp.name)
            pretty_str = pprint.pformat(self.method)
            tmp.write(pretty_str.encode("ascii"))
        finally:
            tmp.close()
            os.system(f"open -a vscodium.app {tmp.name}")
            time.sleep(2)
            os.unlink(tmp.name)
