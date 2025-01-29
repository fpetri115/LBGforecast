# ***DOCUMENTATION IN PROGRESS***
# LBGForecast
Forecasts constraints on cosmological parameters for LSST Lyman-Break Galaxies (LBGs) at z~3-5. Incorporates redshift distribution uncertainties using Stellar Population Synthesis (SPS) simulations.

# Background

# Installation

1. Install FSPS
2. Install prerequisites 
3. Install Speculator fork: `pip install git+https://github.com/fpetri115/speculator.git`
4. git clone repo

# Performing the Forecast
### 1. Sample SPS Parameters from Priors

```
mpiexec -n nproc python sample_sps_params.py 100000 real id path mean
```
Uses MPI to sample SPS parameters for 100000 galaxies, for a total of (nproc x real) realisations each. SPS parameters will be saved as sps_id.npy in path/LBGForecast/sps_parameter_samples. To sample the mean prior, set mean=1, otherwise set mean=0 to sample different (nproc x real) prior realisations.

### 2. Simulate photometry 

#### Option 1: Use Emulator with GPU (faster)
```
python sample_photometry.py path id batch_size
```
Generates noiseless photometry for LSST ugriz bands saved as photo_id.npy in path/LBGForecast/photo_samples.

#### Option 2: Use Python FSPS
```
mpiexec -n nproc python simulate_sps.py id path bands
```
Generates noiseless photometry using either LSST ugrizy (bands = "lsst") or HSC grizy (bands = "suprimecam") filters. Photometry saved as sim_photo_id_bands.npy in path/LBGForecast/photo_samples.
### 3. Apply noise to photometry

```
python photo_to_nz.py path id 0
```
Gives redshift samples for u-, g- and r-dropouts

### 4. PCA Approximation
### 5. Forecast Cosmological Constraints
Marginalise over redshift distribution uncertainties
