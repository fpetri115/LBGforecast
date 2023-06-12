import numpy as np
import matplotlib.pyplot as plt

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


def normalise(nzs, z_space):
        """
        Normalises a redshift distribution. (SLOW)
        -------------------------------------
        Parameters:
        nz - Array containing redshift distribution
        z_space - redshift grid nz is plotted over

        Returns: 
        normalised redshift distribution

        """

        nzs_norm = []
        for nz in nzs:
            area = np.trapz(nz, z_space)
            nzs_norm.append(nz / area)

        return np.array(nzs_norm)

def interloper_fraction(nzs, z_space, z_cut=1.5):

        """
        Given a LBG redshift distribution, cuts nz and z=1.5, and integrates over both peaks
        to get interloper fraction f = number of galaxies @ < z=1.5/number of galaxies @ > z=1.5 (SLOW)
        -------------------------------------------------------------------------------------------
        Parameters:
        nz - Array containing redshift distribution

        Returns:
        f - Interloper Fraction
        """

        nzs = normalise(nzs, z_space)

        # find index in z_space where z>=1.5
        index = 0
        for i in z_space:
            if i >= z_cut:
                break
            index += 1
        
        # divide z_space into two regions, one for lbg, one for interlopers
        z_space_low = z_space[:index]
        z_space_high = z_space[index:]
        
        f_list = []
        for nz in nzs:

            no_int = np.trapz(nz[:index], z_space_low)
            no_lbg = np.trapz(nz[index:], z_space_high)

            f = no_int / (no_lbg + no_int)
            f_list.append(f)

        return np.array(f_list)

def plot_nzs(nzs, z_space, alpha=0.025):
        """
        plots nzs overlaid on one graph
        ---------------------------------------
        Parameters:
        nzs - array of nzs (output of _pca methods and _data methods)

        Returns:
        None

        """
        fig = plt.subplots(1, 1, figsize=(20, 10))
        no = len(nzs)

        i = 0
        while i < no:
            plt.plot(z_space, nzs[i], c="k", alpha=alpha)
            i += 1