import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from scipy.stats import dirichlet

def dirichlet_prior(agebins, alpha, mass_norm):
    """Calculates non-parametric SFH given a Dirichlet
    prior in fsps format.

    :param agebins: 
        An array of bin edges, log(yrs).  This method assumes that the
        upper edge of one bin is the same as the lower edge of another bin.
        ndarray of shape ``(nbin, 2)``
    
    :param alpha:
        Float describing the concentration parameter for a symmetric
        Dirirchlet distribution
    
    :param mass_norm:
        Total stellar mass formed across all age bins (float) in solar
        masses

    :returns tabulatedsfh:
        Output SFH in tabulated fsps format
    
    :returns masses:
        Mass formed in each age bin in solar masses

    """
    nbins = len(agebins)
    alphas = np.ones(nbins)*alpha
    fractions = dirichlet(alphas).rvs(size=1).reshape((nbins,))
    masses = frac_to_masses(mass_norm, fractions, agebins)
    tabulatedsfh = convert_sfh(agebins, masses, epsilon=1e-4)
    
    return tabulatedsfh, masses

def convert_sfh(agebins, mformed, epsilon=1e-4, maxage=None):
        """Given arrays of agebins and formed masses with each bin, calculate a
        tabular SFH.  The resulting time vector has time points either side of
        each bin edge with a "closeness" defined by a parameter epsilon.

        :param agebins:
            An array of bin edges, log(yrs).  This method assumes that the
            upper edge of one bin is the same as the lower edge of another bin.
            ndarray of shape ``(nbin, 2)``

        :param mformed:
            The stellar mass formed in each bin.  ndarray of shape ``(nbin,)``

        :param epsilon: (optional, default 1e-4)
            A small number used to define the fraction time separation of
            adjacent points at the bin edges.

        :param maxage: (optional, default: ``None``)
            A maximum age of stars in the population, in yrs.  If ``None`` then the maximum
            value of ``agebins`` is used.  Note that an error will occur if maxage
            < the maximum age in agebins.

        :returns time:
            The output time array for use with sfh=3, in Gyr.  ndarray of shape (2*N)

        :returns sfr:
            The output sfr array for use with sfh=3, in M_sun/yr.  ndarray of shape (2*N)

        :returns maxage:
            The maximum valid age in the returned isochrone.
        """
        #### create time vector
        agebins_yrs = 10**agebins.T
        dt = agebins_yrs[1, :] - agebins_yrs[0, :]
        bin_edges = np.unique(agebins_yrs)
        if maxage is None:
            maxage = agebins_yrs.max()  # can replace maxage with something else, e.g. tuniv
        t = np.concatenate((bin_edges * (1.-epsilon), bin_edges * (1+epsilon)))
        t.sort()
        t = t[1:-1] # remove older than oldest bin, younger than youngest bin
        fsps_time = maxage - t

        #### calculate SFR at each t
        sfr = mformed / dt
        sfrout = np.zeros_like(t)
        sfrout[::2] = sfr
        sfrout[1::2] = sfr  # * (1+epsilon)

        return (fsps_time / 1e9)[::-1], sfrout[::-1], maxage / 1e9

def zred_to_agebins(zred=0.0, agebins=[], **extras):
    """Set the nonparameteric SFH age bins depending on the age of the universe
    at ``zred``. The first bin is not altered and the last bin is always 15% of
    the upper edge of the oldest bin, but the intervening bins are evenly
    spaced in log(age).

    Parameters
    ----------
    zred : float
        Cosmological redshift.  This sets the age of the universe.

    agebins :  ndarray of shape ``(nbin, 2)``
        The SFH bin edges in log10(years).

    Returns
    -------
    agebins : ndarray of shape ``(nbin, 2)``
        The new SFH bin edges.
    """
    tuniv = cosmo.age(zred).value * 1e9
    tbinmax = tuniv * 0.85
    ncomp = len(agebins)
    agelims = list(agebins[0]) + np.linspace(agebins[1][1], np.log10(tbinmax), ncomp-2).tolist() + [np.log10(tuniv)]
    return np.array([agelims[:-1], agelims[1:]]).T

def frac_to_masses(total_mass, fracs, agebins):
    t = np.diff(10**agebins, axis=-1)[:, 0]
    m_n = []
    n = 0
    for f_n in fracs:
        m_n.append((t[n]*f_n)/np.sum(t*fracs))
        n+=1
    return np.asarray(m_n)*total_mass

def frac_to_sfr(fracs, agebins, total_mass):
    t_n = np.diff(10**agebins, axis=-1)[:, 0]
    m_n = frac_to_masses(total_mass, fracs, agebins)/total_mass

    sfr_n = total_mass*(m_n/t_n)

    return sfr_n 

def nonpar_recent_sfr(masses, agebins, sfr_period=0.1):

    ages = 10**(agebins - 9)
    # fractional coverage of the bin by the sfr period
    ft = np.clip((sfr_period - ages[:, 0]) / (ages[:, 1] - ages[:, 0]), 0., 1)
    mformed = (ft * masses).sum(axis=-1)
    return mformed / (sfr_period * 1e9)

def mwa(sfr, agebins, total_mass_formed):
    """mass-weighted age, vectorized
    """
    ages = 10**(agebins)
    dtsq = (ages[:, 1]**2 - ages[:, 0]**2) / 2
    mwa = (dtsq * sfr).sum() / total_mass_formed
    return mwa / 1e9

def non_parametric_sfh(agebins, massformed, alpha):
    """Sample non parametric star formation histories from symmetric
    Dirichlet prior

    Parameters
    -------------------
    :param agebins:
    An array of bin edges, log(yrs).  This method assumes that the
    upper edge of one bin is the same as the lower edge of another bin.
    ndarray of shape ``(nbin, 2)``

    :param massformed:
    An array of masses, in Solar Masses. Should be shape (nsamples,), where
    nsamples is the number of sfhs to calculate 

    :alpha:
    Concentration parameter for symmetric Dirichlet prior 

    Returns
    :sfhs:
    An array of star formation histories of shape (nsamples, nbins)
    -------------------
    """
    nbins = len(agebins)
    nsamples = massformed.shape[0]
    alphas = np.ones(nbins)*alpha
    fractions = dirichlet(alphas).rvs(size=nsamples).reshape((nsamples, nbins))
    n = 0
    sfhs = []
    while(n < nsamples):
        sfhs.append(frac_to_sfr(fractions[n, :], agebins, massformed[n]))
        n+=1
    return np.array(sfhs)

def tau_model(tau, t):
    return np.exp(-t/tau)

def dpl(tau, a, b, t):
    return ((t/tau)**(a) + (t/tau)**(-b))**(-1)

def normed_sfh(logtau, loga, logb, t):

    sfh = dpl(10**logtau, 10**loga, 10**logb, t)
    
    #if sfh is very small inside age of galaxy, replace with uniform
    if((sfh < 1e-30).all()):
        sfh = np.ones_like(t)

    normed_sfh = sfh/np.trapz((10**9)*sfh, t)

    return normed_sfh

def plot_sfh(sfh, t):

    plt.figure(figsize=(10,5))
    plt.plot(t, sfh)
    plt.xlabel("Time Since the Beginning of the Universe [$\mathrm{Gyr}$]",
               fontsize=12)
    plt.ylabel("Star Formation Rate [$\mathrm{M}_{\odot}\mathrm{yr}^{-1}$]",
               fontsize=12)
    
    plt.tick_params(axis="x", width = 2, labelsize=12*0.8)
    plt.tick_params(axis="y", width = 2, labelsize=12*0.8)