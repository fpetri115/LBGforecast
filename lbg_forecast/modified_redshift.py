# Module to define redshift distributions we can differentiate through
from abc import ABC
from abc import abstractmethod

import jax.numpy as np
from jax.tree_util import register_pytree_node_class

from jax_cosmo.jax_utils import container
from jax_cosmo.scipy.integrate import simps

steradian_to_arcmin2 = 11818102.86004228
path = "./"

__all__ = ["smail_nz", "kde_nz", "delta_nz"]


class redshift_distribution(container):
    def __init__(self, *args, gals_per_arcmin2=1.0, zmax=10.0, **kwargs):
        """Initialize the parameters of the redshift distribution"""
        self._norm = None
        self._gals_per_arcmin2 = gals_per_arcmin2
        super(redshift_distribution, self).__init__(*args, zmax=zmax, **kwargs)

        self.u_npca_components = np.load(path+"/4pca_data/npca_components_u.npy")
        self.g_npca_components = np.load(path+"/4pca_data/npca_components_g.npy")
        self.r_npca_components = np.load(path+"/4pca_data/npca_components_r.npy")

        self.u_npca_mean = np.load(path+"/4pca_data/npca_mean_u.npy")
        self.g_npca_mean = np.load(path+"/4pca_data/npca_mean_g.npy")
        self.r_npca_mean = np.load(path+"/4pca_data/npca_mean_r.npy")

        self.z_grid = np.load(path+"/4pca_data/z_grid.npy")
        self.len_z_grid = self.z_grid.shape[0]

    @abstractmethod
    def pz_fn(self, z):
        """Un-normalized n(z) function provided by sub classes"""
        pass

    def __call__(self, z):
        """Computes the normalized n(z)"""
        if self._norm is None:
            self._norm = simps(lambda t: self.pz_fn(t), 0.0, self.config["zmax"], 256)
        return self.pz_fn(z) / self._norm

    @property
    def zmax(self):
        return self.config["zmax"]

    @property
    def gals_per_arcmin2(self):
        """Returns the number density of galaxies in gals/sq arcmin
        TODO: find a better name
        """
        return self._gals_per_arcmin2

    @property
    def gals_per_steradian(self):
        """Returns the number density of galaxies in steradian"""
        return self._gals_per_arcmin2 * steradian_to_arcmin2

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        children = (self.params, self._gals_per_arcmin2)
        aux_data = self.config
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, gals_per_arcmin2 = children
        return cls(*args, gals_per_arcmin2=gals_per_arcmin2, **aux_data)


@register_pytree_node_class
class smail_nz(redshift_distribution):
    """Defines a smail distribution with these arguments
    Parameters:
    -----------
    a:

    b:

    z0:

    gals_per_arcmin2: number of galaxies per sq arcmin
    """

    def pz_fn(self, z):
        a, b, z0 = self.params
        return z**a * np.exp(-((z / z0) ** b))

@register_pytree_node_class
class u_dropout(redshift_distribution):
    """
    n-component PCA redshift distribution for u dropouts
    -------------------------------------------------------------------

    """

    def pz_fn(self, z):

        z = np.atleast_1d(z)

        pca_components, pca_mean = self.u_npca_components, self.u_npca_mean
        nz_params = self.params[0] #coeffs

        index = np.abs(np.reshape(z, (z.shape[0], 1)) - np.tile(self.z_grid, (z.shape[0], 1))).argmin(axis=1)

        pca_component_i = pca_components[:, index]
        pca_mean_i = pca_mean[index]

        component_sum = np.sum(nz_params*pca_component_i.T, axis=1)
        vec = (component_sum + pca_mean_i)**2

        return vec

@register_pytree_node_class
class g_dropout(redshift_distribution):
    """
    n-component PCA redshift distribution for u dropouts
    -------------------------------------------------------------------

    """

    def pz_fn(self, z):

        z = np.atleast_1d(z)

        pca_components, pca_mean = self.g_npca_components, self.g_npca_mean
        nz_params = self.params[0] #coeffs

        index = np.abs(np.reshape(z, (z.shape[0], 1)) - np.tile(self.z_grid, (z.shape[0], 1))).argmin(axis=1)

        pca_component_i = pca_components[:, index]
        pca_mean_i = pca_mean[index]

        component_sum = np.sum(nz_params*pca_component_i.T, axis=1)
        vec = (component_sum + pca_mean_i)**2
        
        return vec
    
@register_pytree_node_class
class r_dropout(redshift_distribution):
    """
    n-component PCA redshift distribution for u dropouts
    -------------------------------------------------------------------

    """

    def pz_fn(self, z):

        z = np.atleast_1d(z)

        pca_components, pca_mean = self.r_npca_components, self.r_npca_mean
        nz_params = self.params[0] #coeffs

        index = np.abs(np.reshape(z, (z.shape[0], 1)) - np.tile(self.z_grid, (z.shape[0], 1))).argmin(axis=1)

        pca_component_i = pca_components[:, index]
        pca_mean_i = pca_mean[index]

        component_sum = np.sum(nz_params*pca_component_i.T, axis=1)
        vec = (component_sum + pca_mean_i)**2
        
        return vec

@register_pytree_node_class
class histogram_nz(redshift_distribution):
    """Histogram redshift distribution for flexible parameterisation.

    Parameters:
    -----------
    bin_heights: array where elements are histogram bin heights

    bin_edges: array containing edges of each bin and should be
    strictly increasing.

    gals_per_arcmin2: number of galaxies per sq arcmin
    """

    def __init__(self, *args, **kwargs):
        """Initialize the parameters of the redshift distribution"""
        super(histogram_nz, self).__init__(*args, **kwargs)

        if len(self.params[0]) + 1 != len(self.params[1]):
            raise Exception(
                "Number of bin edges must equal the number of bins plus one."
            )

        self._norm = np.sum(self.params[0] * np.diff(self.params[1]))

    def pz_fn(self, z):
        # parameters
        bin_heights = self.params[0]
        bin_edges = self.params[1]

        n_edges = len(bin_heights)
        zgrid_size = len(z)

        # arrays containing lower and upper walls of bins respectively
        zmins = bin_edges[:-1]
        zmaxs = bin_edges[1:]

        # reshape into column vectors
        bin_heights_T = bin_heights.reshape(1, len(bin_heights)).T
        z_T = z.reshape(1, zgrid_size).T

        # build matrices to find whether redshifts lie above each zmin and below each zmax
        zmin_mat = np.repeat(zmins.reshape(1, n_edges), zgrid_size, axis=0)
        zmax_mat = np.repeat(zmaxs.reshape(1, n_edges), zgrid_size, axis=0)
        step_n = np.where(zmin_mat <= z_T, 1.0, 0.0)
        step_p = np.where(zmax_mat > z_T, 1.0, 0.0)

        # find redshifts that lie inside bin adjacent bin edges (i.e. inside bins)
        comparison_mat = np.where(step_n == step_p, 1.0, 0.0)

        return np.dot(comparison_mat, bin_heights_T).flatten()


@register_pytree_node_class
class delta_nz(redshift_distribution):
    """Defines a single plane redshift distribution with these arguments
    Parameters:
    -----------
    z0:
    """

    def __init__(self, *args, **kwargs):
        """Initialize the parameters of the redshift distribution"""
        super(delta_nz, self).__init__(*args, **kwargs)
        self._norm = 1.0

    def pz_fn(self, z):
        z0 = self.params[0]  # editted here
        return np.where(z == z0, 1.0, 0)


@register_pytree_node_class
class kde_nz(redshift_distribution):
    """A redshift distribution based on a KDE estimate of the nz of a
    given catalog currently uses a Gaussian kernel.
    TODO: add more if necessary

    Parameters:
    -----------
    zcat: redshift catalog
    weights: weight for each galaxy between 0 and 1

    Configuration:
    --------------
    bw: Bandwidth for the KDE

    Example:
    nz = kde_nz(redshift_catalog, w, bw=0.1)
    """

    def _kernel(self, bw, X, x):
        """Gaussian kernel for KDE"""
        return (1.0 / np.sqrt(2 * np.pi) / bw) * np.exp(
            -((X - x) ** 2) / (bw**2 * 2.0)
        )

    def pz_fn(self, z):
        # Extract parameters
        zcat, weight = self.params[:2]
        w = np.atleast_1d(weight)
        q = np.sum(w)
        X = np.expand_dims(zcat, axis=-1)
        k = self._kernel(self.config["bw"], X, z)
        return np.dot(k.T, w) / (q)


@register_pytree_node_class
class systematic_shift(redshift_distribution):
    """Implements a systematic shift in a redshift distribution
    TODO: Find a better name for this

    Arguments:
    redshift_distribution
    mean_bias
    """

    def pz_fn(self, z):
        parent_pz, bias = self.params[:2]
        return parent_pz.pz_fn(np.clip(z - bias, 0))


@register_pytree_node_class
class gauss_nz(redshift_distribution):

    def pz_fn(self, z):
        mu, sigma = self.params
        A = 1/(np.sqrt(2*np.pi*sigma**2))
        return A * np.exp(-0.5 * ( ((z-mu) ** 2)/(sigma ** 2)) )


@register_pytree_node_class
class gauss_mixture(redshift_distribution):

    def pz_fn(self, z):
    
        mu1, sig1, mu2, sig2, mu3, sig3 = self.params
        
        pzu = (1/(np.sqrt(2*np.pi*sig1 ** 2))) * np.exp(-0.5 * ( ((z-mu1) ** 2)/(sig1 ** 2)) )
        pzg = (1/(np.sqrt(2*np.pi*sig2 ** 2))) * np.exp(-0.5 * ( ((z-mu2) ** 2)/(sig2 ** 2)) )
        pzr = (1/(np.sqrt(2*np.pi*sig3 ** 2))) * np.exp(-0.5 * ( ((z-mu3) ** 2)/(sig3 ** 2)) )
        
        return (1/2)*pzu + (1/2)*pzg + (1/3)*pzr