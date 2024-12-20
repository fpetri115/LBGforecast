import numpy as np

LSST_AREA_DEG2 = 16000
DEG2_TO_ARCMIN2 = 3600
LSST_AREA_ARCMIN2 = LSST_AREA_DEG2*DEG2_TO_ARCMIN2

def interlopers(samples, cutoff):
    nint = len(np.where(samples<cutoff)[0])
    ntot = len(samples)
    return (nint/ntot)*100