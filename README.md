# sFWRD: a Standardized Fire Weather Research Database

![logos](https://github.com/user-attachments/assets/a4589c3d-26f2-4384-903d-5b6bfc77e744)
![code_screenshots](https://github.com/user-attachments/assets/2e66457e-a7c7-458f-8c71-1132203ecf33)

The Standardized Fire Weather Research Database (sFWRD) project intends to reduce barriers for fire weather research across all technical skill levels. 

Downloadable, analysis-ready, pre-standardized historical fire weather data for 6 reanalysis products (NCEP, NARR, ERA5, NAM12, CONUS404, HRRR), retrospective subseasonal forecast data for UFS (prototype versions 4-8), and select components of SubX, are available at TBD.  

The sFWRD Toolkit the open-source code repository which contains: 
-	Source code to expand the database to meet other variable or time range needs
o	Please note, that due to recent events, some datasets and locations for the ‘Data Grab’ process may be temporarily unavailable. Processing codes should work as expected. 
-	Visualization codes that can be employed with the sFWRD data 
-	Additional tools developed for additional processing and testing such as: downscaling and harmonic analysis 

This project was funded by NOAA WPO Grant #NA23OAR4590383. The sFWRD data and code repository should be cited using the following publication:

	Besong-Cowan et al., The Standardized Fire Weather Research Database, In Preparation.


# Core Features: 

![data_flow](https://github.com/user-attachments/assets/8e8f7c69-78e0-4865-8d4c-492815d5b406)

![variable_metadata](https://github.com/user-attachments/assets/94bf8137-4d86-4b1a-8162-29e81dff74e8)

![spatial_resolution](https://github.com/user-attachments/assets/859227e8-96f2-4c11-b50f-9767067b4789)

![example_maps](https://github.com/user-attachments/assets/15434519-2107-4042-bf43-277c72ce94db)


# Documentation:

Each script contains in-code comments and explanations for transparency and ease of use. 

In addition, the Data Documentation folder contains:
-	Toolkit
o	Data Flow: description of each code in the repository, order of operation to implement mandatory/optional code segments, file tree structure, file naming conventions used throughout 
-	Datasets
o	README’s: raw data origin location, citation, meta data (units, variable names, NaN value, etc.) for data processed through the code bundle 
o	Variable Lookup Table: crosswalk of origin dataset variable name and derivation indicator to the sFWRD standardized naming convention, abbreviation, units, and availability 


# Installation:

This Python code bundle is designed to run using Jupyter notebooks. The following Python packages are primarily utilized: calendar, cartopy, dask, datetime, glob, h5py, math, matplotlib, metpy, netCDF4, numpy, os, pandas, shutil, warnings, xarray, zarr

A YAML file (sfwrd_environment.yml) is provided so that the environment used to develop all sFWRD related code can be replicated.

The ‘Main Toolkit’ should be downloaded, in full, with the relative folder locations retained due to inter-code dependencies. 

The sFWRD codes do not require additional installation once Python, Jupyter notebooks, and the listed packages are established. Updates will be required for pathnames.  
