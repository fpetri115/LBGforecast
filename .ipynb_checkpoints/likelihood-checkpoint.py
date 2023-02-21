import jax
from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit 

import jax.numpy as jnp
import jax.scipy as jsci

import jax_cosmo as jc
from jax_cosmo import Cosmology
from jax_cosmo import probes
from jax_cosmo.angular_cl import angular_cl
from jax_cosmo.angular_cl import noise_cl
from jax_cosmo.angular_cl import gaussian_cl_covariance
from jax_cosmo.angular_cl import gaussian_cl_covariance_and_mean
from jax import jacfwd, jacrev

from modified_redshift import u_dropout
from modified_redshift import g_dropout
from modified_redshift import r_dropout

from modified_bias import custom_bias

from angular_power import cl_theory
from angular_power import cl_data
from angular_power import compare_cls
from angular_power import define_cosmo

def get_cosmo_params(cosmo): 
    return jnp.array([cosmo.sigma8, cosmo.Omega_c, cosmo.Omega_b, cosmo.h, cosmo.n_s, cosmo.w0])

def cosmo_params_to_obj(params):
    cosmo = Cosmology(sigma8=params[0], Omega_c=params[1], 
                      Omega_b=params[2], h=params[3], n_s=params[4], 
                      w0=params[5], Omega_k=0., wa=0.)
    
    return cosmo

def pack_params(cosmo_params, b_lbg, b_int):
    return jnp.hstack((cosmo_params, b_lbg, b_int))

def unpack_params(params):
    cosmo_params = params[:6]
    blbg = params[6]
    bint = params[7]
    
    return cosmo_params, blbg, bint

class Likelihood():
    
    def __init__(self):
        """
        P - N(z) covariance of PCA parameters
        C - Data covariance (includes cosmic variance + cut sky)
        ----------------------------------------------------------
        _mean_vec_u, _mean_vec_g, _mean_vec_r - mean of pca coefficients
        _cov_u, _cov_g, _cov_r - covariance matrix of pca coefficients
        
        """
        print("Initialising likelihood")
        
        self._mean_vec_u = jnp.load("4pca_means_u.npy")
        self._mean_vec_g = jnp.load("4pca_means_g.npy")
        self._mean_vec_r = jnp.load("4pca_means_r.npy")
        
        self._cov_u = jnp.load("4pca_cov_u.npy")
        self._cov_g = jnp.load("4pca_cov_g.npy")
        self._cov_r = jnp.load("4pca_cov_r.npy")
        
        self._npca = len(self._mean_vec_u)
        
        zero_block = jnp.zeros((self._npca, self._npca))
        self.P = jnp.block([[self._cov_u, zero_block, zero_block],
                            [zero_block, self._cov_g, zero_block],
                            [zero_block, zero_block, self._cov_r]])
        self._inv_P = jnp.linalg.inv(self.P)
        
        self._z_cut = 1.5
        self._ell = jnp.arange(200, 1000, 1)
        self._fsky = 0.4
        
        self._b_lbg = 2
        self._b_int = 1
        seed = 100
        
        self.nz_params_mean = jnp.hstack((self._mean_vec_u, self._mean_vec_g, self._mean_vec_r))
        self._cosmo_fid = define_cosmo()
        
        #Generate mock data
        mean_cl, covmat = cl_data(self._cosmo_fid, self.nz_params_mean, 
                                  self._b_lbg, self._b_int, self._ell, self._fsky, seed)
        self.cl_mean = mean_cl
        
        #data covariance
        self.C = covmat
        self._inv_C = jnp.linalg.inv(self.C)
        
        #jacobian
        self._jacobian = jax.jit(jacfwd(cl_theory, argnums=1))
        
        print("Initialisation Complete")
    
    def logL(self, params):
        """marginalised likelihood"""
        
        cosmo_params, blbg, bint = unpack_params(params)
        cosmo = cosmo_params_to_obj(cosmo_params)
        
        nz_params = self.nz_params_mean
        
        T = self._jacobian(cosmo, nz_params, blbg, bint, self._ell)
        Tt = jnp.transpose(T)
        Cinv = self._inv_C
        Pinv = self._inv_P
        P = self.P
        CM = self.C + T@P@Tt
        CMinv = jnp.linalg.inv(CM)
        
        t = cl_theory(cosmo, nz_params, blbg, bint, self._ell)
        c = self.cl_mean
        
        dt = c - t
        dt_T = jnp.transpose(dt)
        
        prefactor = 1/jnp.sqrt(jnp.linalg.det(Tt@Cinv@T + Pinv))
        X = -0.5*dt_T@CMinv@dt
        
        return jnp.log(prefactor) + X
    
    def logLgauss(self, params):
        """Gaussian likelihood for n(z) fixed at mean value"""
        
        cosmo_params, blbg, bint = unpack_params(params)
        cosmo = cosmo_params_to_obj(cosmo_params)
        
        nz_params = self.nz_params_mean
        
        Cinv = self._inv_C
        
        t = cl_theory(cosmo, nz_params, blbg, bint, self._ell)
        c = self.cl_mean
        
        dt = c - t
        dt_T = jnp.transpose(dt)
        
        X = -0.5*dt_T@Cinv@dt
        
        return X
    
    def plot_data_cls(self):
        """Plot mock data used for inference that was initalised with class"""
        
        #data initilaised in lhood
        data_cl = self.cl_mean
        
        #theory vector evaluated at mean redshift distribution
        nz_params = self.nz_params_mean
        cosmo = self._cosmo_fid
        blbg = self._b_lbg
        bint = self._b_int
        theory_cl = cl_theory(cosmo, nz_params, blbg, bint, self._ell)
        
        #plot together
        compare_cls(data_cl, theory_cl, self._ell, figure_size=(15,10), fontsize=18)