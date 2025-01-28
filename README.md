# ***DOCUMENTATION IN PROGRESS***
# LBGForecast
Forecasts constraints on cosmological parameters for LSST Lyman-Break Galaxies (LBGs) at z~3-5. Incorporates redshift distribution uncertainties using Stellar Population Synthesis (SPS) simulations.

# Background

# Installation

1. Install FSPS
2. Install prerequisites 
3. Install Speculator
4. git clone repo

# Performing the Forecast
### 1. Sample SPS Parameters from Priors

```
mpiexec -n nproc python sample_sps_params.py 100000 real id path 0
```
Uses MPI to sample SPS parameters for 100000 galaxies, for a total of nproc*real realisations. SPS parameters will be saved as sps_id.npy in path/LBGForecast/sps_parameter_samples. 

### 2. Simulate photometry using emulator

```
python sample_photometry.py path id batch_size
```
Generates noiseless photometry for LSST ugriz bands saved as photo_id.npy in path/LBGForecast/photo_samples

### 3. Apply noise to photometry

```
python photo_to_nz.py path id 0
```
Gives redshift samples for u-, g- and r-dropouts

### 4. PCA Approximation
### 5. Forecast Cosmological Constraints
Marginalise over redshift distribution uncertainties
