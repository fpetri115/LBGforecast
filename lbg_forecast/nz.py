import numpy as np

def load_redshift_distributions():
    """
    Loads simulated u, g, r dropouts and returns them as a list of numpy arrays.
    Each element of returned list is an array of redshift distributions, where
    every row contains a seperate distribution.

    """

    nzus = np.load("lbg_forecast/nzus.npy")
    nzgs = np.load("lbg_forecast/nzgs.npy")
    nzrs = np.load("lbg_forecast/nzrs.npy")


    if (
        nzus.shape != nzgs.shape
        or nzus.shape != nzrs.shape
        or nzgs.shape != nzrs.shape
    ):
        
        raise Exception("All data must have the same dimensions")

    return [nzus, nzgs, nzrs]


def normalise(self, nz, z_space):
        """
        Normalises a redshift distribution.
        -------------------------------------
        Parameters:
        nz - Array containing redshift distribution
        z_space - redshift grid nz is plotted over

        Returns: 
        normalised redshift distribution

        """
        area = np.trapz(nz, z_space)

        return nz / area

def interloper_fraction(nz, z_space, z_cut=1.5):

        """
        Given a LBG redshift distribution, cuts nz and z=1.5, and integrates over both peaks
        to get interloper fraction f = number of galaxies @ < z=1.5/number of galaxies @ > z=1.5
        -------------------------------------------------------------------------------------------
        Parameters:
        nz - Array containing redshift distribution

        Returns:
        f - Interloper Fraction
        """

        nz = normalise(nz)

        # find index in z_space where z>=1.5
        index = 0
        for i in z_space:
            if i >= z_cut:
                break
            index += 1

        # divide z_space into two regions, one for lbg, one for interlopers
        z_space_low = z_space[:index]
        z_space_high = z_space[index:]

        no_int = np.trapz(nz[:index], z_space_low)
        no_lbg = np.trapz(nz[index:], z_space_high)

        f = no_int / (no_lbg + no_int)

        return f