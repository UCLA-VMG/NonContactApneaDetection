# mmproc

**Contributing Author**: [Adnan Armouti](https://github.com/adnan-armouti).

This library processes raw multimodal (image, rf, audio) data into an ML-ready format.

All code currently under development; please see disclaimer at the end of this README. The "main" branch has been tested on two datasets, while the "low_light" branch is optimized for a single low light self-collected dataset hierarchy.

## Python Installation

**(1) Recommended Installation**

Install Python. Currently we support Python 3.8 only. To download Python, visit https://www.python.org/downloads/. Note that the Python website defaults to 32-bit interpreters, so if you want a 64-bit version of Python you have to click into the specific release version. For ease of installation, we recommend installing python through anaconda distribution, via https://www.anaconda.com/products/distribution

**(2) PATH**

(Optional) Set the PATH environment variable for your Python installation. This may have been done automatically as part of installation, but to do this manually you have to open Environment Variables through the following:

My Computer > Properties > Advanced System Settings > Environment Variables

Add your Python installation location to your PATH variable. For example, if you installed Python at C:\Python38\, you would add the following entry to the PATH variable:

C:\Python38\<rest_of_path>

Note that the anaconda distribution does this for you automatically.

**(3) Configure Installation**

From a command line, run the following commands to update and install dependencies for your associated Python version:

```
>> <python version> -m ensurepip
>> <python version> -m pip install --upgrade pip numpy matplotlib
```

The full list of pip packages can be found in the pip section of the requirements.yaml file, in the configs folder. If you installed python via Anaconda, simply run this command to create an entirely new python environment: 

```
>> conda env create -f requirements.yaml
```

Optionally, you may also update a pre-existing python==3.8 environment via this command: 

```
>> conda env update -f requirements.yaml
```

<hr /> 

## mmproc Installation

Navigate to the mmproc installation directory that you have admin privileges/write permission for, such as C:\Users\Public\Documents\mmproc and execute the following command:

```
>> pip install -e .
```

<hr /> 

## mmproc Usage Guide

This code was developed for use through the mmproc/config/config.yml file. This file currently contains examples that may be adapted as necessary. A detailed PDF guide will be uploaded in the near future for ease of use. Please note: you may use any of the functions defined in this library in any external script or project by simply:

```
>> import mmproc
```

For example:
```
>> import mmproc.utils.utils_read as mmproc_read
>> data = mmproc_read.read_pkl("PATH")
```

<hr /> 

## Current Functionality

Low-level functions are defined in the utils folder, and called by classes defined in the data folder.

**(1) Image Data**
Read, Write, Display, Crop, Resize Spatial, Resize Temporal, Normalize, Bin, Demosaic.

**(2) RF Data**
Read, Write, Display, Range Resolution, Velocity Resolution, Angle Resolution.

**(3) Signal Data**
Read, Write, Display, Interpolate, Crop.

**(4) File Data**
Copy, Delete, Rename, Interpolate, Crop.

<hr /> 

## Functionality in Development

**(1) Audio Data**
Read, Write, Display, Interpolate, Crop, Normalize.

<hr /> 

## Disclaimer

This codebase is still in development, and highly specific to our self-collected dataset hierarcht structure. Use of this code on other datasets may produce bugs, despite the author's best attempt to generalize wherever possible via asbtracted and object oriented programming.

<hr /> 




