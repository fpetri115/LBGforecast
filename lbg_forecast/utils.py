import numpy as np

LSST_AREA_DEG2 = 18000
DEG2_TO_ARCMIN2 = 3600
RAD_TO_DEG = 180/np.pi

FULL_SKY_STERAD = 4*np.pi
FULL_SKY_ARCMIN2 = FULL_SKY_STERAD*(RAD_TO_DEG**2)*DEG2_TO_ARCMIN2
FULL_SKY_DEG2 = FULL_SKY_STERAD*(RAD_TO_DEG**2)

#ones sky contains FULL_SKY_DEG2 degs

#LSST_AREA_ARCMIN2 = LSST_AREA_DEG2*DEG2_TO_ARCMIN2
#LSST_AREA_FRACTION = LSST_AREA_ARCMIN2/FULL_SKY_ARCMIN2

def interlopers(samples, cutoff):
    nint = len(np.where(samples<cutoff)[0])
    ntot = len(samples)
    return (nint/ntot)*100