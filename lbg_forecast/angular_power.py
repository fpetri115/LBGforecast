import jax
from jax.config import config

config.update("jax_enable_x64", True)
from jax import jit
import jax.numpy as jnp

import jax_cosmo as jc
from jax_cosmo import Cosmology

from jax_cosmo.redshift import delta_nz

from lbg_forecast import modified_probes
from jax_cosmo import probes
from jax_cosmo.angular_cl import angular_cl
from lbg_forecast.modified_angular_cl import noise_cl
from lbg_forecast.modified_angular_cl import gaussian_cl_covariance_and_mean

from lbg_forecast.modified_bias import custom_bias
from lbg_forecast.modified_bias import constant_linear_bias

from lbg_forecast.modified_redshift import u_dropout
from lbg_forecast.modified_redshift import g_dropout
from lbg_forecast.modified_redshift import r_dropout

from lbg_forecast.modified_redshift import histogram_nz
from lbg_forecast.modified_angular_cl import angular_cl as new_cl


from functools import partial

import matplotlib.pyplot as plt


def define_cosmo():
    """
    Define a cosmology in jax-cosmo (Planck 2015 results)

    """
    return jc.Planck15()


def z_space():
    """
    Redshift space grid for redshift distributions

    """
    return jnp.arange(0, 7, 0.01)


@jit
def cl_theory(cosmo, nz_params, b_lbg, b_int, ell):
    """
    Calculates theory vector for Likelihood. Computes angular cls
    and cross correlations of u, g, r dropouts with two component bias.
    --------------------------------------------------------------------
    Parameters:
    cosmo - JAX-COSMO cosmology object containing cosmological parameters
    nz_params -
    b_int - Interloper bias (linear)
    b_lbg - LBG bias (linear)
    ell - Spherical harmonic scale list. Gives range of ells to plot cls over
    ----------------------------------------------------------------------
    Returns:
    Concatenated angular power spectra of length 6*len(ell) giving auto+cross
    spectra, with poisson noise

    """
    n = 4
    z_cut = 1.5

    nz_u = u_dropout(nz_params[:n], gals_per_arcmin2=1)#1
    nz_g = g_dropout(nz_params[n : 2 * n], gals_per_arcmin2=1)#1
    nz_r = r_dropout(nz_params[2 * n : 3 * n], gals_per_arcmin2=0.1)#0.1

    redshift_distributions = [nz_u, nz_g, nz_r]

    bias = custom_bias(b_int, b_lbg, z_cut)

    tracers = [probes.NumberCounts(redshift_distributions, bias)]

    signal = new_cl(cosmo, ell, tracers)
    noise = noise_cl(ell, tracers)
    total_cl = signal + noise

    return jnp.hstack(total_cl)

@jit
def cl_theory_CMB(cosmo, nz_params, b_lbg, b_int, ell):
    """
    Calculates theory vector for Likelihood. Computes angular cls
    and cross correlations of u, g, r dropouts with two component bias.
    --------------------------------------------------------------------
    Parameters:
    cosmo - JAX-COSMO cosmology object containing cosmological parameters
    nz_params -
    b_int - Interloper bias (linear)
    b_lbg - LBG bias (linear)
    ell - Spherical harmonic scale list. Gives range of ells to plot cls over
    ----------------------------------------------------------------------
    Returns:
    Concatenated angular power spectra of length 6*len(ell) giving auto+cross
    spectra, with poisson noise

    """
    n = 4
    z_cut = 1.5

    surface_of_last_scattering = delta_nz(1100., gals_per_arcmin2 = 1e10) 

    nz_u = u_dropout(nz_params[:n], gals_per_arcmin2=1)#1
    nz_g = g_dropout(nz_params[n : 2 * n], gals_per_arcmin2=1)#1
    nz_r = r_dropout(nz_params[2 * n : 3 * n], gals_per_arcmin2=0.1)#0.1

    redshift_distributions = [nz_u, nz_g, nz_r]

    bias = custom_bias(b_int, b_lbg, z_cut)

    cosmo_probes = [probes.NumberCounts(redshift_distributions, bias),
                    modified_probes.WeakLensing([surface_of_last_scattering])]

    signal = angular_cl(cosmo, ell, cosmo_probes)
    noise = noise_cl(ell, cosmo_probes)
    total_cl = signal + noise

    return jnp.hstack(total_cl)

#@partial(jit, static_argnums=7)
def cl_data(cosmo, nz_params, b_lbg, b_int, ell, f_sky, seed, ncls):
    """
    Genrates Mock LBG lustering angular power spectra data. Gives
    u, g, r-dropout clustering plus cross correlations. Gaussian noise
    is added to simulate cosmic variance, which is also scaled by
    sky fraction.
    --------------------------------------------------------------------
    Parameters:
    cosmo - JAX-COSMO cosmology object containing cosmological parameters
    nz_params -
    b_int - Interloper bias (linear)
    b_lbg - LBG bias (linear)
    ell - Spherical harmonic scale list. Gives range of ells to plot cls over
    ----------------------------------------------------------------------
    Returns:
    Concatenated angular power spectra of length 6*len(ell) giving auto+cross
    spectra, with poisson noise, cosmic variance plus contribution from cut sky

    """
    n = 4
    z_cut = 1.5

    nz_u = u_dropout(nz_params[:n], gals_per_arcmin2=1)
    nz_g = g_dropout(nz_params[n : 2 * n], gals_per_arcmin2=1)
    nz_r = r_dropout(nz_params[2 * n : 3 * n], gals_per_arcmin2=0.1)

    redshift_distributions = [nz_u, nz_g, nz_r]

    bias = custom_bias(b_int, b_lbg, z_cut)

    tracers = [probes.NumberCounts(redshift_distributions, bias)]

    signal, cov = gaussian_cl_covariance_and_mean(
        cosmo, ell, tracers, f_sky=f_sky, sparse=False
    )

    noise = jnp.hstack(noise_cl(ell, tracers))
    total_cl = signal + noise

    total_cl = add_noise(total_cl, cov, seed, len(ell), ncls)

    return total_cl, cov

#@jit
def cl_data_CMB(cosmo, nz_params, b_lbg, b_int, ell, f_sky, seed, ncls):
    """
    Genrates Mock LBG lustering angular power spectra data. Gives
    u, g, r-dropout clustering plus cross correlations. Gaussian noise
    is added to simulate cosmic variance, which is also scaled by
    sky fraction.
    --------------------------------------------------------------------
    Parameters:
    cosmo - JAX-COSMO cosmology object containing cosmological parameters
    nz_params -
    b_int - Interloper bias (linear)
    b_lbg - LBG bias (linear)
    ell - Spherical harmonic scale list. Gives range of ells to plot cls over
    ----------------------------------------------------------------------
    Returns:
    Concatenated angular power spectra of length 6*len(ell) giving auto+cross
    spectra, with poisson noise, cosmic variance plus contribution from cut sky

    """
    n = 4
    z_cut = 1.5

    surface_of_last_scattering = delta_nz(1100., gals_per_arcmin2 = 1e10) 

    nz_u = u_dropout(nz_params[:n], gals_per_arcmin2=1)
    nz_g = g_dropout(nz_params[n : 2 * n], gals_per_arcmin2=1)
    nz_r = r_dropout(nz_params[2 * n : 3 * n], gals_per_arcmin2=0.1)

    redshift_distributions = [nz_u, nz_g, nz_r]

    bias = custom_bias(b_int, b_lbg, z_cut)

    cosmo_probes = [probes.NumberCounts(redshift_distributions, bias),
                    modified_probes.WeakLensing([surface_of_last_scattering])]

    signal, cov = gaussian_cl_covariance_and_mean(
        cosmo, ell, cosmo_probes, f_sky=f_sky, sparse=False
    )

    noise = jnp.hstack(noise_cl(ell, cosmo_probes))
    total_cl = signal + noise

    total_cl = add_noise(total_cl, cov, seed, len(ell), ncls)

    return total_cl, cov

def generate_uncorr_normal(seed, ell_length, ncls):
    """
    Generate uncorrelated gaussian random numbers
    """
    tot_plots = ncls*ncls - sum(jnp.arange(0, ncls, 1))

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    return jax.random.normal(subkey, shape=(1, ell_length * tot_plots))[0]


def generate_corr_num(cov, seed, ell_length, ncls):
    """
    Performs Cholesky Decomposition to generate correlated random
    numbers using an array of gaussian distributed, uncorrelated
    random numbers in generate_uncorr_normal() and a desired
    covariance matrix
    -------------------------------------------------------------
    Parameters:
    cov - Desired covariance matrix of generated random numbers
    seed - Random number seed
    ell_length - Spherical harmonic ell range
    ------------------------------------------------------------
    Returns correlated random numbers with covariance = cov
    """
    L = jnp.linalg.cholesky(cov)
    a = generate_uncorr_normal(seed, ell_length, ncls)
    x = L @ a

    return x


def add_noise(cl, cov, seed, ell_length, ncls):
    """
    Adds noise from generate_corr_num() to angular
    power specturm

    """
    x = generate_corr_num(cov, seed, ell_length, ncls)
    return cl + x


def plot_ncls(cls_theory, ell, figure_size, fontsize, ncls):
    """
    Plots auto and cross power spectra in a triangle plot.
    ----------------------------------------------------------
    Parameters:
    cls_theory - Concatenated theory vector from cl_theory()
    ell - Spherical harmonic scale list. Gives range of ells to plot cls over
    figure_size, fontsize - plotting

    """
    tot_plots = ncls*ncls - sum(jnp.arange(0, ncls, 1))

    fig, axes = plt.subplots(ncls, ncls, figsize=figure_size)
    cl_list = jnp.split(cls_theory, tot_plots)

    i = 0
    j = 0
    k = 0
    while j < ncls:
        i = 0
        while i < ncls:
            ax = axes[i][j]
            if i >= j:
                ax.plot(ell, cl_list[k])
                # ax.set_xscale("log")
                ax.set_yscale("log")
                k += 1
            else:
                ax.set_visible(False)

            # plotting labels
            if i == 2 and j == 0:
                ax.set_ylabel("$C_{\ell}$", fontsize=fontsize)
            if i == 2 and j == 1:
                ax.set_xlabel("$\ell$", fontsize=fontsize)

            i += 1

        j += 1


def compare_cls(cl1, cl2, ell, figure_size, fontsize, ncls):
    """
    Plots two sets of cls on one plot in order to compare
    ----------------------------------------------------------
    Parameters:
    cl1 - Concatenated cl vector 1
    cl2 - Concatenated cl vector 2
    ell - Spherical harmonic scale list. Gives range of
          ells to plot cls over

    (figure_size, fontsize - plotting)

    """
    tot_plots = ncls*ncls - sum(jnp.arange(0, ncls, 1))

    fig, axes = plt.subplots(ncls, ncls, figsize=figure_size)
    cl1_list = jnp.split(cl1, tot_plots)
    cl2_list = jnp.split(cl2, tot_plots)

    i = 0
    j = 0
    k = 0
    while j < ncls:
        i = 0
        while i < ncls:
            ax = axes[i][j]
            if i >= j:
                ax.plot(ell, cl1_list[k])
                ax.plot(ell, cl2_list[k])
                # ax.set_xscale("log")
                ax.set_yscale("log")
                k += 1
            else:
                ax.set_visible(False)

            # plotting labels
            if i == 1 and j == 0:
                ax.set_ylabel("$C_{\ell}$", fontsize=fontsize)
            if i == 2 and j == 1:
                ax.set_xlabel("$\ell$", fontsize=fontsize)

            i += 1

        j += 1


@jit
def cl_hat(cosmo, bin_heights, bin_edges, ell):
    """function to test hist nz"""

    nz = [histogram_nz(bin_heights, bin_edges)]
    bias = constant_linear_bias(1.0)
    tracers = [modified_probes.NumberCounts(nz, bias)]

    signal = new_cl(cosmo, ell, tracers)

    return signal.flatten()
