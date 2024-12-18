import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax_cosmo import Cosmology
from jax import jacfwd

from lbg_forecast.angular_power import cl_theory
from lbg_forecast.angular_power import cl_theory_CMB
from lbg_forecast.angular_power import cl_data
from lbg_forecast.angular_power import cl_data_CMB
from lbg_forecast.angular_power import compare_cls
from lbg_forecast.angular_power import define_cosmo
from lbg_forecast.angular_power import pk

from lbg_forecast.modified_likelihood import gaussian_log_likelihood
from lbg_forecast.modified_likelihood import marginalised_log_likelihood

def get_cosmo_params(cosmo):
    return jnp.array(
        [cosmo.sigma8, cosmo.Omega_c, cosmo.Omega_b, cosmo.h, cosmo.n_s, cosmo.w0]
    )


def cosmo_params_to_obj(params):
    cosmo = Cosmology(
        sigma8=params[0],
        Omega_c=params[1],
        Omega_b=params[2],
        h=params[3],
        n_s=params[4],
        w0=params[5],
        Omega_k=0.0,
        wa=0.0,
    )

    return cosmo

def pack_params(cosmo_params, b_lbg, b_int):
    return jnp.hstack((cosmo_params, b_lbg, b_int))


def unpack_params(params):
    cosmo_params = params[:6]
    blbg = params[6]
    bint = params[7]

    return cosmo_params, blbg, bint


class Likelihood:
    def __init__(self, path):
        """
        P - N(z) covariance of PCA parameters
        C - Data covariance (includes cosmic variance + cut sky)
        ----------------------------------------------------------
        _mean_vec_u, _mean_vec_g, _mean_vec_r - mean of pca coefficients
        _cov_u, _cov_g, _cov_r - covariance matrix of pca coefficients

        """
        print("Initialising likelihood")

        self._mean_vec_u = jnp.load(path+"/4pca_data/4pca_means_u.npy")
        self._mean_vec_g = jnp.load(path+"/4pca_data/4pca_means_g.npy")
        self._mean_vec_r = jnp.load(path+"/4pca_data/4pca_means_r.npy")

        self._cov_u = jnp.load(path+"/4pca_data/4pca_cov_u.npy")
        self._cov_g = jnp.load(path+"/4pca_data/4pca_cov_g.npy")
        self._cov_r = jnp.load(path+"/4pca_data/4pca_cov_r.npy")

        self._npca = len(self._mean_vec_u)

        zero_block = jnp.zeros((self._npca, self._npca))
        self.P = jnp.block(
            [
                [self._cov_u, zero_block, zero_block],
                [zero_block, self._cov_g, zero_block],
                [zero_block, zero_block, self._cov_r],
            ]
        )
        self._inv_P = jnp.linalg.inv(self.P)

        self._z_cut = 1.5
        self._ell = jnp.arange(200, 1000, 1)
        self._fsky = 0.4
        seed = 100

        self._b_lbg_u = 3
        self._b_lbg_g = 5
        self._b_lbg_r = 7

        self._b_int_u = 3
        self._b_int_g = 5
        self._b_int_r = 7

        self._bias_params = jnp.array([self._b_lbg_u,
                                       self._b_lbg_g,
                                       self._b_lbg_r,
                                       self._b_int_u,
                                       self._b_int_g,
                                       self._b_int_r
        ])

        self.nz_params_mean = jnp.hstack(
            (self._mean_vec_u, self._mean_vec_g, self._mean_vec_r)
        )
        self._cosmo_fid = define_cosmo()

        _o_m = self._cosmo_fid.Omega_c + self._cosmo_fid.Omega_b
        _s8 = self._cosmo_fid.sigma8*jnp.sqrt(_o_m/0.3)

        self._derived_params = jnp.array([_o_m, _s8])

        # Generate mock data
        mean_cl, covmat = cl_data_CMB(
            self._cosmo_fid,
            self.nz_params_mean,
            self._bias_params,
            self._ell,
            self._fsky,
            seed,
            ncls = 4,
        )
        self.cl_mean = mean_cl

        # data covariance
        self.C = covmat
        self._inv_C = jnp.linalg.inv(self.C)

        # jacobian
        self._jacobian = jax.jit(jacfwd(cl_theory_CMB, argnums=1))
        self.T = self._jacobian(self._cosmo_fid, self.nz_params_mean,
                                 self._bias_params, self._ell)
        
        self.Cm = self.C + self.T @ self.P @ self.T.T

        ####For use with Fisher:
        cosmo_params = get_cosmo_params(self._cosmo_fid)
        self._combined_params = jnp.concatenate((cosmo_params, 
                                                 self._bias_params, 
                                                 self._derived_params))

        print("Initialisation Complete")


    def _mu_vec(self, params, inds):
        """Reduced theory vector for fisher forecast"""

        combined_params = self._combined_params
        params_ind = 0
        final_index = 0
        for i in inds:
            combined_params = combined_params.at[i].set(params[params_ind])
            params_ind += 1
            final_index = i

        ####Stuff for W&W
        #norm_diff = pk(self._cosmo_fid, 1/8, 0)/pk(self._cosmo_fid, 1/8, 2.6)
        #cosmo_params = cosmo_params.at[0].set(params[0]*jnp.sqrt(norm_diff)) #sigma8 at z=2.6
        ####

        bias_params = combined_params[6:9]
        derived_params = combined_params[9:]
        nz_params = self.nz_params_mean

        cosmo = cosmo_params_to_obj(combined_params[:6])
    
        return cl_theory_CMB(cosmo, nz_params, bias_params, self._ell)
    
    def _mu_vec_deriv(self, params, inds):
        """Reduced theory vector for fisher forecast"""

        combined_params = self._combined_params
        params_ind = 0
        final_index = 0
        for i in inds:
            combined_params = combined_params.at[i].set(params[params_ind])
            params_ind += 1
            final_index = i

        bias_params = combined_params[6:9]
        derived_params = combined_params[9:]
        nz_params = self.nz_params_mean


        o_m = derived_params[0]
        s8 = derived_params[1]
        combined_params = combined_params.at[0].set(s8/jnp.sqrt(o_m/0.3))
        combined_params = combined_params.at[1].set(o_m - combined_params[2])

        cosmo = cosmo_params_to_obj(combined_params[:6])
    
        return cl_theory_CMB(cosmo, nz_params, bias_params, self._ell)

    def logL(self, params):
        """marginalised likelihood"""

        cosmo_params, bias_params = unpack_params(params)
        cosmo = cosmo_params_to_obj(cosmo_params)

        nz_params = self.nz_params_mean

        T = self.T
        C = self.C
        P = self.P

        t = cl_theory_CMB(cosmo, nz_params, bias_params, self._ell)
        c = self.cl_mean

        return marginalised_log_likelihood(c, t, C, P, T)

    def logLgauss(self, params):
        """Gaussian likelihood for n(z) fixed at mean value"""

        cosmo_params, bias_params = unpack_params(params)
        cosmo = cosmo_params_to_obj(cosmo_params)

        nz_params = self.nz_params_mean

        cov = self.C

        t = cl_theory_CMB(cosmo, nz_params, bias_params, self._ell)
        c = self.cl_mean

        return gaussian_log_likelihood(c, t, cov, include_logdet=False)
    
    def fisher(self, params, inds):

        inv_cov = jnp.linalg.inv(self.C)
        jac_at_mean = jax.jit(jax.jacfwd(self._mu_vec, argnums=0))
        dmudp = jac_at_mean(params, inds)

        F = dmudp.T@inv_cov@dmudp

        return F
    
    def fisher_marg(self, params, inds):

        inv_cov = jnp.linalg.inv(self.Cm)
        jac_at_mean = jax.jit(jax.jacfwd(self._mu_vec, argnums=0))
        dmudp = jac_at_mean(params, inds)

        F = dmudp.T@inv_cov@dmudp

        return F
    
    def fisher_deriv(self, params, inds):

        inv_cov = jnp.linalg.inv(self.C)
        jac_at_mean = jax.jit(jax.jacfwd(self._mu_vec_deriv, argnums=0))
        dmudp = jac_at_mean(params, inds)

        F = dmudp.T@inv_cov@dmudp

        return F
    
    def fisher_marg_deriv(self, params, inds):

        inv_cov = jnp.linalg.inv(self.Cm)
        jac_at_mean = jax.jit(jax.jacfwd(self._mu_vec_deriv, argnums=0))
        dmudp = jac_at_mean(params, inds)

        F = dmudp.T@inv_cov@dmudp

        return F

    def plot_data_cls(self):
        """Plot mock data used for inference that was initalised with class"""

        # data initilaised in lhood
        data_cl = self.cl_mean

        # theory vector evaluated at mean redshift distribution
        nz_params = self.nz_params_mean
        cosmo = self._cosmo_fid
        bias_params = self._bias_params
        theory_cl = cl_theory_CMB(cosmo, nz_params, bias_params, self._ell)

        # plot together
        compare_cls(data_cl, theory_cl, self._ell, figure_size=(15, 10), fontsize=18)
