# regridFV.py

`regridFV.py` is a Python script designed to regrid FVCom data files
according to specified parameters. It allows users to set input and
output details, choose variables of interest, specify the resolution
for output, and select an interpolation method.

## Requirements

To install all the required dependencies:

```
$ conda env create -f environment_simple.yaml
```

## Usage

The script is executed from the command line with several options to specify the input parameters:

- `-i`, `--input`: Path to the input file
- `-o`, `--output`: Directory where the output files will be stored
- `-r`, `--resolution`: Desired resolution for the output data in meters
- `-v`, `--variables`: Specify the variables of interest separated by colons (e.g., `salinity:u:v`)
- `-p`, `--prefix`: Prefix for the output filenames
- `-m`, `--interp`: Interpolation method to be used (e.g., `linear`, `nearest`)
- `-b`, `--bbox`: Bounding box defined as minLat:minLon:maxLat:maxLon

Example call:

```
$ python regridFV.py --input=in/proc_sms_0025.nc -m linear --output=out --resolution=100 --prefix=test --variables=salinity:zeta --bbox=43.266:4.32:43.7138:4.93
```


## Todo List

- Parallelization of `gen_4dvar`, `gen_3dvar` function calls
- Parallelization of timestep generation
- Adding attributes to NetCDF variables
