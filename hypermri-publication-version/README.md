# hypermri 🧲

Schilling-AG (Technical University of Munich) Python module to work with pre-clinical Paravision 6/7 Bruker MRI/MRS data.
# Important Notice
This is an early pre-release version of the hypermri repository, which will be linked here as soon as its released officially.
This version will not be maintained and provides a snapshot at time of publication of #insert DOI# (early 2023).

## Installation

The creation of a new environment is a must!
```bash
$ git clone 'path-to-Github'

$ cd 'folder'

$ conda create -n hypermri_env python==3.10.10 && conda activate hypermri_env

(hypermri_env)$ pip install -r dev-requirements.txt

(hypermri_env)$ pip install -e ".[dev]"
```

## Usage
```python
# BrukerDir attempts to smartly load in each scan, that is, if a specific sequence-class
# is configured, that specifig class is used to load in the bruker scan.
# Note that every sequence class inherits from the base class BrukerExp. This is also
# the class the smartloader falls back to,  in case it does not recognise the scan.

from hypermri import BrukerDir

# load in the complete experiment directory using the BrukerDir class
# Note: in windows you might have to use r"..." when providing the path

scans = BrukerDir("folderpath/containing/all/scans/from/one/experiment")

# to get an overview over all scans at any time, use:
scans.display()
```
