## Disclaimer
This repository is a snapshot of processing code used for the publication "Parahydrogen-Polarized [1-13C]Pyruvate for Reliable and Fast Preclinical Metabolic Magnetic Resonance Imaging"(https://doi.org/10.1002/advs.202303441).
This code is not maintained, its use is purely to provide the current state of data analysis for reproducibility.

## Overview
This code consists of two python modules that are used by four data analysis jupyter notebooks.
### Modules
1. hypermri: This module is found in hypermri-publication-version and can be installed as described below. Its use
is for loading, analyzing and processing preclinical MRI and MRS data from a Bruker MRI operated by Paravision 7.
2. MagriProc.py: This module is used for loading, analyzing and processing NMR data from a Spinsolve tabletop spectrometer 
by Magritek.
### Analysis notebooks
All these notebooks should be opened using jupyter lab version 7.
1. Example_T1_T2_analysis.ipynb

    This notebook shows exemplary how T1 and T2 values were analyzed for this study.
2. Example_Polarization_level_calculation.ipynb

    This notebook shows how the polarization level was calculated.
3. Example_Perfusion_Measurement.ipynb

    This notebook shows how in vivo measurements using a bssfp sequence were analyzed for
    perfusion experiments.
4. Example_Metabolism_Measurement.ipynb
   
    This notebook shows how in vivo measurements using a bssfp sequence were analyzed
    for tumor metabolism measurements.



# Useage
1. Clone this repository, i.e. using Windows Powershell or Mac Terminal:
```bash
$ git clone https://github.com/QuE-MRT/2023_TUMKRI_NVIS_PHIPvsDNPInVivoStudy
```
2. Navigate to the folder containing the hypermri package, create an environment and install the package
```bash
$ cd hypermri_publication_version
$ conda  conda create -n hypermri_env python==3.10.10
$ conda activate hypermri_env
(hypermri_env)$ pip install -r dev-requirements.txt
(hypermri_env)$ pip install -e ".[dev]"
```
3. Change to the main folder and open jupyter lab (version>=3.5.3):
```bash
(hypermri_env)$ cd ..
(hypermri_env)$ jupyter lab
```
4. Test data is available upon request and should be placed in the same directory as the notebooks.

# Contributing
This is a static repository and contributing is not possible.

# Licensing
This repository is licensed under Apache 2.0.

However code from two external repositories (pyNMR by Benno Meier and brukerMRI by Joerg Döpfert) is integrated
into some functions. 
Therefore a folder with their respective Licenses is included here as well and their code is cited within the packages.
