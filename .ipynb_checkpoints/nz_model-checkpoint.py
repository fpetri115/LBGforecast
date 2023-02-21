import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_redshift_distributions():
    
    """
    Loads simulated u, g, r dropouts and returns them as a list of numpy arrays.
    Each element of returned list is an array of redshift distributions, where
    every row contains a seperate distribution.

    """
    
    nzus = np.load("nzus.npy")
    nzgs = np.load("nzgs.npy")
    nzrs = np.load("nzrs.npy")
    
    return [nzus, nzgs, nzrs]

def perform_2pca(bin_vals):
    
    """
    Given a set of simulated n(z)'s, given by bin_vals, get 2-component PCA.
    ------------------------------------------------------------------------
    Parameters:
    bin_vals - Can be found using u_data(), g_data(), r_data() methods in NzModel.
    ------------------------------------------------------------------------
    Returns: 
    bin_pca - bin_pca[:,0] gives 1st PCA component coeffecients, 
    bin_pca[:,1] gives 2nd etc. of data.
    pca_components_ - List of PCA components (2 of them) (minus pca.mean_)
    pca.mean_ - mean of components
    """
    
    bin_vals_sqrt = np.sqrt(bin_vals)
    
    scaler = StandardScaler()

    scaler.fit(bin_vals_sqrt)
    bin_scaled = scaler.transform(bin_vals_sqrt)
    
    pca = PCA(n_components=2)
    pca.fit(bin_vals_sqrt)
    bin_pca = pca.transform(bin_vals_sqrt)
    
    return [bin_pca, pca.components_, pca.mean_]

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
    
    return [bin_pca, pca.components_, pca.mean_]

def gauss_2pca(pca_data, n):
    
    """
    Gaussian approximation of 2-component PCA given in perform_2pca(). 
    Gives gaussian distributed PCA coeffecients.
    ---------------------------------------------------------------------
    Paraneters:
    n - number of samples
    """
    bin_pca, pca_components, pca_mean = pca_data
    
    pca_coeff1 = bin_pca[:,0]
    pca_coeff2 = bin_pca[:,1]
    
    gauss_pca_coeffs = np.random.multivariate_normal((np.mean(pca_coeff1), np.mean(pca_coeff2)), np.cov(pca_coeff1, pca_coeff2), n)
    
    pca_nzs = []
    
    i = 0
    while(i < n):
        func = ((pca_components[0])*gauss_pca_coeffs[i][0] + (pca_components[1])*gauss_pca_coeffs[i][1] + pca_mean)**2
        pca_nzs.append(func)
        i+=1
    
    return np.array(pca_nzs)

def gauss_npca(pca_data, n_s):
    
    """
    Gaussian approximation of n-component PCA given in perform_npca(). 
    Gives gaussian distributed PCA coeffecients.
    ---------------------------------------------------------------------
    Paraneters:
    n_s - number of samples
    n - pca components
    """
    bin_pca, pca_components, pca_mean = pca_data
    pca_coeffs = []
    
    #pca_components
    n = len(pca_components)
    
    i = 0
    while(i < n):
        pca_coeffs.append(bin_pca[:,i])
        i+=1
        
    pca_coeffs = np.array(pca_coeffs)
    pca_means = []
    
    i = 0
    while(i < n):
        pca_means.append(np.mean(pca_coeffs[i]))
        i+=1
        
    
    gauss_pca_coeffs = np.random.multivariate_normal(pca_means,
                                                     np.cov(pca_coeffs),
                                                     n_s)
    
    pca_nzs = []
    
    i = 0
    while(i < n_s):
        
        func = 0
        j = 0
        while(j < n):
            func += pca_components[j]*gauss_pca_coeffs[i][j]
            j+=1
    
        pca_nzs.append((func + pca_mean)**2)
        i+=1
    
    return np.array(pca_nzs) 
  
def pca_mean_cov(pca_data):
    """
    Calculate PCA coeffecient mean and covariance from PCA
    decomposition of simulated data
    -----------------------------------------------------
    Returns means and covariance
    """
    
    bin_pca, pca_components, pca_mean = pca_data
    pca_coeffs = []
    
    #pca_components
    n = len(pca_components)
    
    i = 0
    while(i < n):
        pca_coeffs.append(bin_pca[:,i])
        i+=1
        
    pca_coeffs = np.array(pca_coeffs)
    pca_means = []
    
    i = 0
    while(i < n):
        pca_means.append(np.mean(pca_coeffs[i]))
        i+=1
        
    pca_means = np.array(pca_means)
    pca_cov = np.cov(pca_coeffs)
    
    return pca_means, pca_cov
    

class NzModel:
    
    """
    This module will contain all information regarding the redshift distributions
    
    """
    
    def __init__(self, z_space):

        nzs = load_redshift_distributions()

        if(nzs[0].shape != nzs[1].shape or 
           nzs[0].shape != nzs[2].shape or 
           nzs[1].shape != nzs[2].shape):
            
            raise Exception("All data must have the same dimensions")

        self._nzus = nzs[0]
        self._nzgs = nzs[1]
        self._nzrs = nzs[2]

        self._n_simulations = self._nzus.shape[0]
        self._z_length = self._nzus.shape[1]
        
        self._z_space = z_space
        
        #used for splitting nzs into interloper and lbg component
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
    
    def plot_nzs(self, nzs, alpha=0.05):
        """
        plots nzs overlaid on one graph
        ---------------------------------------
        Parameters: 
        nzs - array of nzs (output of _pca methods and _data methods)
        
        Returns:
        None
        
        """
        fig = plt.subplots(1, 1, figsize = (20, 10))
        no = len(nzs)
        
        i = 0
        while(i <  no):
            plt.plot(self._z_space, nzs[i], c="k", alpha=alpha)
            i+=1
    
    def plot_all_pca(self, n, n_s):
        
        """
        Plots all u, g, r, pca nzs
        ----------------------------------------
        Parameters:
        n - number of nz gaussian-pca samples
        
        Returns:
        None
        
        """
        
        fig = plt.subplots(1, 1, figsize = (20, 10))
        
        unzs = self.u_pca(n, n_s)
        gnzs = self.g_pca(n, n_s)
        rnzs = self.r_pca(n, n_s)
        
        #u
        i = 0
        while(i < n_s):
            plt.plot(self._z_space, unzs[i], c="b", alpha=0.05)
            i+=1
            
        #g
        i = 0
        while(i < n_s):
            plt.plot(self._z_space, gnzs[i], c="g", alpha=0.05)
            i+=1
                   
        #r
        i = 0
        while(i < n_s):
            plt.plot(self._z_space, rnzs[i], c="r", alpha=0.05)
            i+=1
            
    def export_nzs(self, n, n_s):
        """ 
        Exports all gaussian-pca models of u, g, r redhsift distributions into .npy files
        ---------------------------------------------------------------------------------
        Parameters:
        n - Number of nz gaussian-pca samples

        Returns:
        None

        """
        unzs = self.u_pca(n, n_s)
        gnzs = self.g_pca(n, n_s)
        rnzs = self.r_pca(n, n_s)


        np.save("nzus_pca.npy", unzs)
        np.save("nzgs_pca.npy", gnzs)
        np.save("nzrs_pca.npy", rnzs)
        
        np.save("nzus_pca_lbg.npy", self.lbg_component(unzs))
        np.save("nzgs_pca_lbg.npy", self.lbg_component(gnzs))
        np.save("nzrs_pca_lbg.npy", self.lbg_component(rnzs))
        
        np.save("nzus_pca_int.npy", self.interloper_component(unzs))
        np.save("nzgs_pca_int.npy", self.interloper_component(gnzs))
        np.save("nzrs_pca_int.npy", self.interloper_component(rnzs))
    
        
        
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
        
        return nz/self.normalisation(nz)
    
    def normalise_nzs(self, nzs):
        """
        Normalise list/array of redshift distributions given by parameter nzs.
        Returns array of normalised distributions
        
        """
        normalised_nzs = []
        
        for nz in nzs:
            normalised_nzs.append(self.normalise(nz))
        
        return np.array(normalised_nzs)
    
    def lbg_component(self, nzs):
        """
        Find the lbg dropout component of redshift distribution
        -------------------------------------------------------
        Parameters:
        nzs - Array of redshift distributions
        Returns: 
        Array of redshift distributions with only lbg component
        
        """
        
        lbg_nzs = []
        z_cut = []
        
        #sets up z_cut to, which will be multiplied with nz
        for z in self._z_space:
            if(z >= self._z_cut):
                z_cut.append(1)
            else:
                z_cut.append(0)        
        z_cut = np.array(z_cut)
        
        for nz in nzs:
            
            #multiply nz by array, where z<1.5 are multiplied by zero
            lbg_nzs.append(nz*z_cut)
            
        return self.normalise_nzs(lbg_nzs)
    
    def interloper_component(self, nzs):
        """
        Find the interloper component of redshift distribution
        ------------------------------------------------------
        Parameters:
        nzs - Array of redshift distributions
        Returns: 
        Array of redshift distributions with only interloper component
        
        """
        
        int_nzs = []
        z_cut = []
        
        #sets up z_cut to, which will be multiplied with nz
        for z in self._z_space:
            if(z >= self._z_cut):
                z_cut.append(0)
            else:
                z_cut.append(1)       
        z_cut = np.array(z_cut)
        
        for nz in nzs:
            
            #multiply nz by array, where z>=1.5 are multiplied by zero
            int_nzs.append(nz*z_cut)
            
        return self.normalise_nzs(int_nzs)
    
    def interloper_fraction(self, nz):
        """
        Given a LBG redshift distribution, cuts nz and z=1.5, and integrates over both peaks
        to get interloper fraction f = number of galaxies @ < z=1.5/number of galaxies @ > z=1.5
        -------------------------------------------------------------------------------------------
        Parameters:
        nz - Array containing redshift distribution
        
        Returns:
        f - Interloper Fraction 
        """
        
        nz = self.normalise(nz)
        
        #find index in z_space where z>=1.5
        index = 0
        for i in self._z_space:
            if(i >= self._z_cut):
                break
            index+=1
        
        #divide z_space into two regions, one for lbg, one for interlopers
        z_space_low = self._z_space[:index]
        z_space_high = self._z_space[index:]
        

        
        no_int = np.trapz(nz[:index], z_space_low)
        no_lbg = np.trapz(nz[index:], z_space_high)
        
        f = (no_int/(no_lbg+no_int))

        
        return f
    
    def save_4pca_data(self):
        """saves 4-Component PCA related data"""
        
        u_pca_data = perform_npca(self.u_data(), 4)
        g_pca_data = perform_npca(self.g_data(), 4)
        r_pca_data = perform_npca(self.r_data(), 4)
        
        u_pca_means, u_pca_cov = pca_mean_cov(u_pca_data)
        g_pca_means, g_pca_cov = pca_mean_cov(g_pca_data)
        r_pca_means, r_pca_cov = pca_mean_cov(r_pca_data)
        
        np.save("4pca_data/4pca_components_u.npy", u_pca_data[1])
        np.save("4pca_data/4pca_components_g.npy", g_pca_data[1])
        np.save("4pca_data/4pca_components_r.npy", r_pca_data[1])
        
        np.save("4pca_data/4pca_mean_u.npy", u_pca_data[2])
        np.save("4pca_data/4pca_mean_g.npy", g_pca_data[2])
        np.save("4pca_data/4pca_mean_r.npy", r_pca_data[2])
        
        np.save("4pca_data/4pca_means_u.npy", u_pca_means)
        np.save("4pca_data/4pca_means_g.npy", g_pca_means)
        np.save("4pca_data/4pca_means_r.npy", r_pca_means)
        
        np.save("4pca_data/4pca_cov_u.npy", u_pca_cov)
        np.save("4pca_data/4pca_cov_g.npy", g_pca_cov)
        np.save("4pca_data/4pca_cov_r.npy", r_pca_cov)