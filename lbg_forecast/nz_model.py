import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_redshift_distributions(path):
    """
    Loads simulated u, g, r dropouts and returns them as a list of numpy arrays.
    Each element of returned list is an array of redshift distributions, where
    every row contains a seperate distribution.

    """

    nzus = np.load(path+"/redshifts/nzus.npy")
    nzgs = np.load(path+"/redshifts/nzgs.npy")
    nzrs = np.load(path+"/redshifts/nzrs.npy")
    z_grid = np.load(path+"redshifts/z_grid.npy")

    return z_grid, [nzus, nzgs, nzrs]


def perform_npca(bin_vals, n):
    """
    Given a set of simulated n(z)'s, given by bin_vals, get n-component PCA.
    ------------------------------------------------------------------------
    Parameters:
    bin_vals - Can be found using u_data(), g_data(), r_data() methods in NzModel.
    n - number of pca components
    ------------------------------------------------------------------------
    Returns:
    bin_pca - bin_pca[:,0] gives 1st PCA component coeffecients, bin_pca[:,1]
    gives 2nd etc. of data.
    pca_components_ - List of n PCA components (minus pca.mean_)
    pca.mean_ - mean of components
    """

    bin_vals_sqrt = np.sqrt(bin_vals)

    scaler = StandardScaler()

    scaler.fit(bin_vals_sqrt)
    bin_scaled = scaler.transform(bin_vals_sqrt)

    pca = PCA(n_components=n)
    pca.fit(bin_vals_sqrt)
    bin_pca = pca.transform(bin_vals_sqrt)

    return [bin_pca, pca.components_, pca.mean_, pca.explained_variance_ratio_, pca.explained_variance_]


def gauss_npca(pca_data, n_s):
    """
    Gaussian approximation of n-component PCA given in perform_npca().
    Gives gaussian distributed PCA coeffecients.
    ---------------------------------------------------------------------
    Paraneters:
    n_s - number of samples
    n - pca components
    """
    bin_pca, pca_components, pca_mean, pca_explained_var_ratio, pca_explained_var = pca_data
    pca_coeffs = []

    # pca_components
    n = len(pca_components)

    i = 0
    while i < n:
        pca_coeffs.append(bin_pca[:, i])
        i += 1

    pca_coeffs = np.array(pca_coeffs)
    pca_means = []

    i = 0
    while i < n:
        pca_means.append(np.mean(pca_coeffs[i]))
        i += 1

    print(np.diag(np.cov(pca_coeffs)))
    print(pca_explained_var)

    #gauss_pca_coeffs = np.random.multivariate_normal(pca_means, np.cov(pca_coeffs), n_s)
    gauss_pca_coeffs = np.random.multivariate_normal(pca_means, np.diag(pca_explained_var), n_s)



    pca_nzs = []

    i = 0
    while i < n_s:
        func = 0
        j = 0
        while j < n:
            func += pca_components[j] * gauss_pca_coeffs[i][j]
            j += 1

        pca_nzs.append((func + pca_mean) ** 2)
        i += 1

    return np.array(pca_nzs)


def pca_mean_cov(pca_data):
    """
    Calculate PCA coeffecient mean and covariance from PCA
    decomposition of simulated data
    -----------------------------------------------------
    Returns means and covariance
    """

    bin_pca, pca_components, pca_mean, pca_explained_var_ratio, pca_explained_var = pca_data
    pca_coeffs = []

    # pca_components
    n = len(pca_components)

    i = 0
    while i < n:
        pca_coeffs.append(bin_pca[:, i])
        i += 1

    pca_coeffs = np.array(pca_coeffs)
    pca_means = []

    i = 0
    while i < n:
        pca_means.append(np.mean(pca_coeffs[i]))
        i += 1

    pca_means = np.array(pca_means)
    pca_cov = np.diag(pca_explained_var)  #np.cov(pca_coeffs)

    return pca_means, pca_cov


class NzModel:

    """
    This module will contain all information regarding the redshift distributions

    """

    def __init__(self, path):
        z_grid, nzs = load_redshift_distributions(path)

        if (
            nzs[0].shape != nzs[1].shape
            or nzs[0].shape != nzs[2].shape
            or nzs[1].shape != nzs[2].shape
        ):
            raise Exception("All data must have the same dimensions")

        self._nzus = nzs[0]
        self._nzgs = nzs[1]
        self._nzrs = nzs[2]

        self._n_simulations = self._nzus.shape[0]
        self._z_length = self._nzus.shape[1]

        self._z_space = z_grid
        self._factor = 100
        self._plotting_zspace = np.linspace(self._z_space[0], self._z_space[-1], len(self._z_space)*self._factor)


        # used for splitting nzs into interloper and lbg component
        self._z_cut = 1.5

    def simulations_used(self):
        """Access number of simulations used in data"""
        return self._n_simulations

    def z_length(self):
        """Access total number of samples in redshift space each nz is plotted in the data"""
        return self._z_length

    def u_data(self):
        """Access simulated u dropouts"""
        return self._nzus

    def g_data(self):
        """Access simulated g dropouts"""
        return self._nzgs

    def r_data(self):
        """Access simulated r dropouts"""
        return self._nzrs

    def u_pca(self, n, n_s):
        """
        Return a sample (array of n(z)s) of normalised
        u dropout n(z)'s from the gaussian/pca model

        """
        pca_data = perform_npca(self.u_data(), n)
        u_pca = gauss_npca(pca_data, n_s)

        return self.normalise_nzs(u_pca)

    def g_pca(self, n, n_s):
        """
        Return a sample (array of n(z)s) of normalised
        g dropout n(z)'s from the gaussian/pca model

        """
        pca_data = perform_npca(self.g_data(), n)
        g_pca = gauss_npca(pca_data, n_s)

        return self.normalise_nzs(g_pca)

    def r_pca(self, n, n_s):
        """
        Return a sample (array of n(z)s) of normalised
        r dropout n(z)'s from the gaussian/pca model

        """
        pca_data = perform_npca(self.r_data(), n)
        r_pca = gauss_npca(pca_data, n_s)

        return self.normalise_nzs(r_pca)

    def plot_nzs(self, nzs, alpha=0.025):
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
            plt.plot(self._z_space, nzs[i], c="k", alpha=alpha)
            i += 1

    def plot_all_data(self, alpha = 0.1):
        
        fig = plt.subplots(1, 1, figsize=(20, 10))

        u_nzs = self.u_data()
        g_nzs = self.g_data()
        r_nzs = self.r_data()

        p1=99.7
        p2=97.5
        p3=84

        u_mean = np.repeat(np.mean(u_nzs, axis=0), self._factor)
        u_percentile = np.repeat(np.percentile(u_nzs, [100-p1, 100-p2, 100-p3, 50, p3, p2, p1], axis=0), self._factor, axis=1)

        plt.plot(self._plotting_zspace, u_mean, color='blue', ls='--')
        plt.fill_between(self._plotting_zspace, u_percentile[0, :], u_percentile[-1, :], alpha=0.1, color='blue')
        plt.fill_between(self._plotting_zspace, u_percentile[1, :], u_percentile[-2, :], alpha=0.2, color='blue')
        plt.fill_between(self._plotting_zspace, u_percentile[2, :], u_percentile[-3, :], alpha=0.3, color='blue')

        g_mean = np.repeat(np.mean(g_nzs, axis=0), self._factor)
        g_percentile = np.repeat(np.percentile(g_nzs, [100-p1, 100-p2, 100-p3, 50, p3, p2, p1], axis=0), self._factor, axis=1) 

        plt.plot(self._plotting_zspace, g_mean, color='green', ls='-.')
        plt.fill_between(self._plotting_zspace, g_percentile[0, :], g_percentile[-1, :], alpha=0.1, color='green')
        plt.fill_between(self._plotting_zspace, g_percentile[1, :], g_percentile[-2, :], alpha=0.2, color='green')
        plt.fill_between(self._plotting_zspace, g_percentile[2, :], g_percentile[-3, :], alpha=0.3, color='green')

        r_mean = np.repeat(np.mean(r_nzs, axis=0), self._factor)
        r_percentile = np.repeat(np.percentile(r_nzs, [100-p1, 100-p2, 100-p3, 50, p3, p2, p1], axis=0), self._factor, axis=1)

        plt.plot(self._plotting_zspace, r_mean, color='red', ls='-')
        plt.fill_between(self._plotting_zspace, r_percentile[0, :], r_percentile[-1, :], alpha=0.1, color='red')
        plt.fill_between(self._plotting_zspace, r_percentile[1, :], r_percentile[-2, :], alpha=0.2, color='red')
        plt.fill_between(self._plotting_zspace, r_percentile[2, :], r_percentile[-3, :], alpha=0.3, color='red')

        plt.xlabel("$z$", fontsize=22)
        plt.xticks(fontsize=22)

        plt.ylabel("$p(z)$", fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(loc=0, fontsize= 24, frameon=False, ncols=3)

        ax = plt.gca()
        plt.setp(ax.spines.values(), linewidth=3)
        ax.xaxis.set_tick_params(width=3)
        ax.yaxis.set_tick_params(width=3)

        ax.tick_params(direction='in', length=8)

        ax2 = ax.secondary_yaxis("right")
        ax2.tick_params(axis="y", direction="in", length=8, labelright=False)
        plt.setp(ax2.spines.values(), linewidth=3)
        ax2.yaxis.set_tick_params(width=3)

        ax3 = ax.secondary_xaxis("top")
        ax3.tick_params(axis="x", direction="in", length=8, labeltop=False)
        plt.setp(ax3.spines.values(), linewidth=3)
        ax3.xaxis.set_tick_params(width=3)   

    def plot_all_pca(self, n, n_s, alpha=0.01):
        """
        Plots all u, g, r, pca nzs
        ----------------------------------------
        Parameters:
        n - number of nz gaussian-pca samples

        Returns:
        None

        """

        fig = plt.subplots(1, 1, figsize=(20, 10))

        u_nzs = self.u_pca(n, n_s)
        g_nzs = self.g_pca(n, n_s)
        r_nzs = self.r_pca(n, n_s)

        p1=99.7
        p2=97.5
        p3=84

        u_mean = np.repeat(np.mean(u_nzs, axis=0), self._factor)
        u_percentile = np.repeat(np.percentile(u_nzs, [100-p1, 100-p2, 100-p3, 50, p3, p2, p1], axis=0), self._factor, axis=1)

        plt.plot(self._plotting_zspace, u_mean, color='blue', ls='--')
        plt.fill_between(self._plotting_zspace, u_percentile[0, :], u_percentile[-1, :], alpha=0.1, color='blue')
        plt.fill_between(self._plotting_zspace, u_percentile[1, :], u_percentile[-2, :], alpha=0.2, color='blue')
        plt.fill_between(self._plotting_zspace, u_percentile[2, :], u_percentile[-3, :], alpha=0.3, color='blue')

        g_mean = np.repeat(np.mean(g_nzs, axis=0), self._factor)
        g_percentile = np.repeat(np.percentile(g_nzs, [100-p1, 100-p2, 100-p3, 50, p3, p2, p1], axis=0), self._factor, axis=1) 

        plt.plot(self._plotting_zspace, g_mean, color='green', ls='-.')
        plt.fill_between(self._plotting_zspace, g_percentile[0, :], g_percentile[-1, :], alpha=0.1, color='green')
        plt.fill_between(self._plotting_zspace, g_percentile[1, :], g_percentile[-2, :], alpha=0.2, color='green')
        plt.fill_between(self._plotting_zspace, g_percentile[2, :], g_percentile[-3, :], alpha=0.3, color='green')

        r_mean = np.repeat(np.mean(r_nzs, axis=0), self._factor)
        r_percentile = np.repeat(np.percentile(r_nzs, [100-p1, 100-p2, 100-p3, 50, p3, p2, p1], axis=0), self._factor, axis=1)

        plt.plot(self._plotting_zspace, r_mean, color='red', ls='-')
        plt.fill_between(self._plotting_zspace, r_percentile[0, :], r_percentile[-1, :], alpha=0.1, color='red')
        plt.fill_between(self._plotting_zspace, r_percentile[1, :], r_percentile[-2, :], alpha=0.2, color='red')
        plt.fill_between(self._plotting_zspace, r_percentile[2, :], r_percentile[-3, :], alpha=0.3, color='red')

        plt.xlabel("$z$", fontsize=22)
        plt.xticks(fontsize=22)

        plt.ylabel("$p(z)$", fontsize=22)
        plt.yticks(fontsize=22)

        ax = plt.gca()
        plt.setp(ax.spines.values(), linewidth=3)
        ax.xaxis.set_tick_params(width=3)
        ax.yaxis.set_tick_params(width=3)

        ax.tick_params(direction='in', length=8)

        ax2 = ax.secondary_yaxis("right")
        ax2.tick_params(axis="y", direction="in", length=8, labelright=False)
        plt.setp(ax2.spines.values(), linewidth=3)
        ax2.yaxis.set_tick_params(width=3)

        ax3 = ax.secondary_xaxis("top")
        ax3.tick_params(axis="x", direction="in", length=8, labeltop=False)
        plt.setp(ax3.spines.values(), linewidth=3)
        ax3.xaxis.set_tick_params(width=3)

    def normalisation(self, nz):
        """
        Calculates normalisation of a given nz.
        -----------------------------------------
        Parameters:
        nz - Array containing redshift distribution

        Returns:
        norm - Normalisation

        """

        return np.trapz(nz, self._z_space)

    def normalise(self, nz):
        """
        Normalises redshift distribution.
        -------------------------------------
        Parameters:
        nz - Array containing redshift distribution

        Returns: normalised redshift distribution

        """

        return nz / self.normalisation(nz)

    def normalise_nzs(self, nzs):
        """
        Normalise list/array of redshift distributions given by parameter nzs.
        Returns array of normalised distributions

        """
        normalised_nzs = []

        for nz in nzs:
            normalised_nzs.append(self.normalise(nz))

        return np.array(normalised_nzs)

    def save_npca_data(self, npca, path):
        """saves n-Component PCA related data"""

        u_pca_data = perform_npca(self.u_data(), npca)
        g_pca_data = perform_npca(self.g_data(), npca)
        r_pca_data = perform_npca(self.r_data(), npca)

        u_pca_means, u_pca_cov = pca_mean_cov(u_pca_data)
        g_pca_means, g_pca_cov = pca_mean_cov(g_pca_data)
        r_pca_means, r_pca_cov = pca_mean_cov(r_pca_data)

        np.save(path+"/4pca_data/npca_components_u.npy", u_pca_data[1])
        np.save(path+"/4pca_data/npca_components_g.npy", g_pca_data[1])
        np.save(path+"/4pca_data/npca_components_r.npy", r_pca_data[1])

        np.save(path+"/4pca_data/npca_mean_u.npy", u_pca_data[2])
        np.save(path+"/4pca_data/npca_mean_g.npy", g_pca_data[2])
        np.save(path+"/4pca_data/npca_mean_r.npy", r_pca_data[2])

        np.save(path+"/4pca_data/npca_means_u.npy", u_pca_means)
        np.save(path+"/4pca_data/npca_means_g.npy", g_pca_means)
        np.save(path+"/4pca_data/npca_means_r.npy", r_pca_means)

        np.save(path+"/4pca_data/npca_cov_u.npy", u_pca_cov)
        np.save(path+"/4pca_data/npca_cov_g.npy", g_pca_cov)
        np.save(path+"/4pca_data/npca_cov_r.npy", r_pca_cov)

        np.save(path+"/4pca_data/z_grid.npy", self._z_space)
