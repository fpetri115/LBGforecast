
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from scipy import integrate
import emcee
from getdist import plots, MCSamples

def generate_observed_params(nsamples, truth=False):

    if(truth == False):
        logphi1 = np.random.normal([-2.44, -3.08, -4.14], [0.04, 0.05, 0.2], (nsamples, 3))
        logphi2 = np.random.normal([-2.89, -3.29, -3.51], [0.07, 0.06, 0.06], (nsamples, 3))
        logm_star = np.random.normal([10.79, 10.88, 10.84], [0.04, 0.04, 0.08], (nsamples, 3))
        #alpha1 = np.random.uniform(-0.5 , 1, nsamples)
        #alpha2 = np.random.uniform(-2 , -0.5, nsamples)
        alpha1 = np.random.normal(-0.28 , 0.14, nsamples)
        alpha2 = np.random.normal(-1.48 , 0.04, nsamples)
    else:
        logphi1 = np.array([[-2.44, -3.08, -4.14]])
        logphi2 = np.array([[-2.89, -3.29, -3.51]])
        logm_star = np.array([[10.79, 10.88, 10.84]])
        alpha1 = np.array([[-0.28]])
        alpha2 = np.array([[-1.48]])

    return [logphi1, logphi2, logm_star, alpha1, alpha2]

def get_quadratic_coeffs(samples):

    y1 = samples[:, 2]
    y2 = samples[:, 1]
    y3 = samples[:, 0]

    z1 = 3.0
    z2 = 1.6
    z3 = 0.2

    a = ((y3-y1) + ((y2-y1) / (z2-z1)) * (z1-z3)) / ((z3**2-z1**2 + ((z2**2-z1**2) / (z2-z1)) * (z1-z3)))
    b = ((y2-y1) - a * (z2**2-z1**2)) / (z2-z1)
    c = y1-a*z1**2-b*z1

    return np.transpose(np.vstack((a, b, c)))

def mass_function_parameter_curves(z_grid, nsamples, truth):

    observed_mass_function_parameters = generate_observed_params(nsamples, truth)
    redshift_dependent_parameters = observed_mass_function_parameters[:3]
    alphas = observed_mass_function_parameters[3:]

    curves = []
    i = 0
    for param in redshift_dependent_parameters:
        coeffs = get_quadratic_coeffs(param)
        a = np.reshape(coeffs[:, 0], (nsamples, 1))
        b = np.reshape(coeffs[:, 1], (nsamples, 1))
        c = np.reshape(coeffs[:, 2], (nsamples, 1))
        z_grid_tile = np.tile(z_grid, (nsamples, 1))

        logparam = c*np.ones_like(z_grid_tile) + b*z_grid_tile + a*z_grid_tile*z_grid_tile
        i+=1

        curves.append(logparam)

    for aparam in alphas:
        z_grid_tile = np.tile(z_grid, (nsamples, 1))
        alpha_param = np.reshape(aparam, (nsamples, 1))
        alpha_param = alpha_param*np.ones_like(z_grid_tile)
        curves.append(alpha_param)

    return curves

def select_allowed_parameter_curves(z_grid, curves):

    redshift_dependent_curves =  curves[:3]
    alphas = curves[3:]
    selected_curves = []
        
    #prior_bounds = [np.array([np.log10(0.1e-3), np.log10(2.4e-3)]), np.array([-100, -4]), np.array([10, 12])]

    prior_bounds = [np.array([np.log10(0.1e-3), np.log10(0.6e-1)]), np.array([-100, -1]), np.array([10, 12])]
    #prior_bounds = [np.array([np.log10(1e-5), np.log10(1e-3)]), np.array([-100, -4]), np.array([10, 12])]
    i = 0
    for param in redshift_dependent_curves:
        param_df = pd.DataFrame(param, columns=z_grid)


        param_df.drop(param_df[param_df[z_grid[-1]] > prior_bounds[i][1]].index, axis=0, inplace=True)
        param_df.drop(param_df[param_df[z_grid[-1]] < prior_bounds[i][0]].index, axis=0, inplace=True)

        #else:
        #    z_cutoff=3.0
        #    high_z_indexes = np.where(z_grid > z_cutoff)[0]
        #    high_zs = z_grid[high_z_indexes]

            #print(param_df[high_zs])
        #    for z in high_zs:
        #        param_df.drop(param_df[param_df[z] > 2.4e-3].index, axis=0, inplace=True)
        #        param_df.drop(param_df[param_df[z] < 0.01e-3].index, axis=0, inplace=True)


            #param_df.drop(param_df[param_df[z_grid[-1]] < 0.01e-3].index, axis=0, inplace=True)

        selected_curves.append(param_df.to_numpy())
        i+=1

    for aparam in alphas:
        selected_curves.append(aparam)

    return selected_curves

def preload_parameter_curves(nprior_draws, z_grid, init_nsamples, truth=False):
    """preloads a set of logphi1, logphi2, logm*, alpha1, alpha2 curves
    as function of redshift
    """
    if(truth):
        preloaded_curves = mass_function_parameter_curves(z_grid, 1, truth=truth)
    else:
        curves = mass_function_parameter_curves(z_grid, init_nsamples, truth=truth)
        allowed_curves = select_allowed_parameter_curves(z_grid, curves)
        preloaded_curves = sample_allowed_parameter_curves(z_grid, nprior_draws, allowed_curves)

    return preloaded_curves

def draw_parameter_curves(preloaded_curves):
    """samples a single set of curves from preloaded 
    set of logphi1, logphi2, logm*, alpha1, alpha2 curves taken
    from preload_parameter_curves()
    
    """
    nsampled_curves = preloaded_curves[0].shape[0]
    curve_index = np.random.randint(0, nsampled_curves)
    sampled_curves = []
    for param in preloaded_curves:
        sampled_curves.append(param[curve_index, :])
    
    sampled_curves = np.vstack(sampled_curves)

    return sampled_curves

def plot_parameter_curves(z_grid, sampled_curves, log_phi_plot, **kwargs):
    """Plots output of draw_parameter_curves() only
    
    """
    fig, axes = plt.subplots(1, 5)
    fig.set_figheight(10)
    fig.set_figwidth(30)
    ylabels = ['log$_{10}\phi_{1}$', 'log$_{10}\phi_{2}$', 'log$_{10}\mathrm{M}_{*}$',r'$\alpha_{1}$', r'$\alpha_{2}$']

    indx = 0
    while(indx < 5):

        axes[indx].plot(z_grid, sampled_curves[indx, :], c='purple', **kwargs)
        axes[indx].set_xlabel('$z$', fontsize=20)
        axes[indx].set_ylabel(ylabels[indx], fontsize=20)

        indx+=1

def schechter_function(logm, logphi, logm_star, alpha):
    
    return np.log(10)*(10**logphi)*10**((logm-logm_star)*(alpha+1))*np.exp(-10**(logm-logm_star))

def mass_function(z, logm, z_dependence, z_grid):

    z_dependent_parameters = []
    indx = 0
    while(indx < 5):
        z_dependent_parameters.append(np.interp(z, z_grid, z_dependence[indx, :]))
        indx+=1

    logphi1, logphi2, logm_star, alpha1, alpha2 = z_dependent_parameters
    mfunc = schechter_function(logm, logphi1, logm_star, alpha1) + schechter_function(logm, logphi2, logm_star, alpha2)

    if(mfunc < 1e-100):
        mfunc = 1e-100

    return mfunc

def log_n(x, z_dependence, z_grid, v_grid, prior_bounds=[0.0,3.0,9.99,10.01]):

    z, logm = x

    if(z < prior_bounds[0] or z > prior_bounds[1]):
        return -np.inf
    
    if(logm < prior_bounds[2] or logm > prior_bounds[3]):
        return -np.inf
    
    phi = mass_function(z, logm, z_dependence, z_grid)
    volumes = np.interp(z, z_grid, v_grid)
    ngalaxies = phi*volumes

    return np.log(ngalaxies)

def reproduce_plots(nwalkers=50, steps=5000):
    """takes approx 2mins
    """
    #setup
    z_grid = np.linspace(0.0, 3.0, 50)
    v_grid = volume_elements(z_grid)
    preloaded_z_dependent_curves = preload_parameter_curves(1000, z_grid, 1000000, truth=True)
    sampled_curves = draw_parameter_curves(preloaded_z_dependent_curves)
    plot_parameter_curves(z_grid, sampled_curves)

    #sampling
    ndim = 2
    pz = np.random.uniform(1.01, 1.02, (nwalkers, 1))
    plogm = np.random.uniform(9.9999, 10.0001, (nwalkers, 1))
    p0 = np.hstack((pz, plogm))
    prior_bounds=[0.0,3.0,9.99,10.01]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_n, args=[sampled_curves, z_grid, v_grid, prior_bounds])

    state = sampler.run_mcmc(p0, 100)
    sampler.reset()

    sampler.run_mcmc(state, steps)

    #plotting

    samples = sampler.get_chain(flat=True)
    s = MCSamples(samples=samples, names=["z", "logm"])
    plotter = plots.get_subplot_plotter()
    plotter.triangle_plot([s], Filled=True, contour_lws=2)
    print(samples.shape)

    figure, axes = plt.subplots(1,1)
    axes.hist(samples[:, 0], bins=50, density=True)
    plot_observed_nz()

def plot_observed_nz():
    
    z_grid = np.linspace(0.0, 3.0, 100)
    logm_grid = np.linspace(10, 10, 1)
    observed_parameter_curves = mass_function_parameter_curves(z_grid, 1, truth=True)
    mass_func = np.reshape(10**mass_function(logm_grid, observed_parameter_curves), (100,))
    nz = mass_func*volume_elements(z_grid)
    nz = nz/np.trapz(nz, z_grid)

    plt.plot(z_grid, nz)
    plt.xlabel('z')
    plt.ylabel('p(z|logm)')

def sample(nwalkers=50, steps=5000):
    """
    """
    #setup
    z_grid = np.linspace(0.0, 15.0, 50)
    v_grid = volume_elements(z_grid)
    preloaded_z_dependent_curves = preload_parameter_curves(1000, z_grid, 1000000, truth=False)
    sampled_curves = draw_parameter_curves(preloaded_z_dependent_curves)
    plot_parameter_curves(z_grid, sampled_curves)

    #sampling
    ndim = 2
    pz = np.random.uniform(1.01, 1.02, (nwalkers, 1))
    plogm = np.random.uniform(9.9, 10.1, (nwalkers, 1))
    p0 = np.hstack((pz, plogm))
    prior_bounds=[0.0,10.0,7,13]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_n, args=[sampled_curves, z_grid, v_grid, prior_bounds])

    state = sampler.run_mcmc(p0, 100)
    sampler.reset()

    sampler.run_mcmc(state, steps)

    #plotting

    samples = sampler.get_chain(flat=True)
    s = MCSamples(samples=samples, names=["z", "logm"])
    plotter = plots.get_subplot_plotter()
    plotter.triangle_plot([s], Filled=True, contour_lws=2)
    print(samples.shape)

def sample_redshift_mass_prior(nsamples, prior_data, prior_bounds=[0.0,10.0,7,13], plotting=False):
    """
    """

    nwalkers = 100
    steps = 20000
    burnin = 5000
    if(nsamples >= steps*nwalkers - burnin):
        raise Exception("Requesting too many samples")
    
    #setup
    preloaded_z_dependent_curves, z_grid, v_grid = prior_data
    sampled_curves = draw_parameter_curves(preloaded_z_dependent_curves)

    #sampling
    ndim = 2
    pz = np.random.uniform(1.01, 1.02, (nwalkers, 1))
    plogm = np.random.uniform(9.9, 10.1, (nwalkers, 1))
    p0 = np.hstack((pz, plogm))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_n, args=[sampled_curves, z_grid, v_grid, prior_bounds])

    #state = sampler.run_mcmc(p0, 100)
    #sampler.reset()

    sampler.run_mcmc(p0, steps)

    #plotting
    samples = sampler.get_chain(flat=True)
    redshift_samples = samples[:, 0]
    mass_samples = samples[:, 1]
    if(plotting):
        plot_parameter_curves(z_grid, sampled_curves)
        plot_redshift_mass_prior(redshift_samples, mass_samples)

    return redshift_samples[burnin:nsamples+burnin], mass_samples[burnin:nsamples+burnin]

def plot_redshift_mass_prior(redshift_samples, mass_samples):
        
        samples = np.hstack((np.reshape(redshift_samples, (len(redshift_samples), 1)), np.reshape(mass_samples,  (len(mass_samples), 1))))
        print(samples.shape)
        s = MCSamples(samples=samples, names=["z", "logm"])
        plotter = plots.get_subplot_plotter()
        plotter.triangle_plot([s], Filled=True, contour_lws=2)

        fig, ax = plt.subplots(1, 2)
        fig.set_figheight(10)
        fig.set_figwidth(30)
        ax[0].hist(redshift_samples, bins=50)
        ax[0].set_xlabel('z', fontsize=24)
        ax[0].tick_params('x', labelsize=24)
        ax[0].tick_params('y', labelsize=24)
        ax[1].hist(mass_samples, bins=50)
        ax[1].tick_params('x', labelsize=24)
        ax[1].tick_params('y', labelsize=24)
        ax[1].set_xlabel('log(M)', fontsize=24)

def preload_prior_data(zmax=7.0):
    z_grid = np.linspace(0.0, zmax, 100)
    v_grid = volume_elements(z_grid)
    preloaded_z_dependent_curves = preload_parameter_curves(10000, z_grid, 1500000, truth=False)

    return preloaded_z_dependent_curves, z_grid, v_grid


def volume_elements(z_grid):

    dz = z_grid[-1]-z_grid[-2]
    volumes = []
    for z in z_grid:
        volumes.append(volume_element(z, dz))

    return np.array(volumes)

def volume_element(z, dz):
    return cosmo.comoving_volume(z+dz).value - cosmo.comoving_volume(z).value


def plot_mass_function_parameter_curves(z_grid, curves, log_phi_plot, **kwargs):

    fig, axes = plt.subplots(1, 5)
    fig.set_figheight(10)
    fig.set_figwidth(30)
    ylabels = ['$\phi_{1}10^{-3}\mathrm{Mpc}^{-3}\mathrm{dex}^{-1}$', '$\phi_{2}10^{-3}\mathrm{Mpc}^{-3}\mathrm{dex}^{-1}$', 'log$_{10}\mathrm{M}_{*}$',r'$\alpha_{1}$', r'$\alpha_{2}$']
    ylabels_log = ['log$_{10}\phi_{1}$', 'log$_{10}\phi_{2}$', 'log$_{10}\mathrm{M}_{*}$',r'$\alpha_{1}$', r'$\alpha_{2}$']
    plot_no = 0
    for param in curves:
        i = 0
        ncurves = curves[plot_no].shape[0]
        if(log_phi_plot == True or (plot_no != 0 and plot_no != 1)):
            while(i < ncurves):
                axes[plot_no].plot(z_grid, param[i, :], c='purple', **kwargs)
                i+=1
            axes[plot_no].set_ylabel(ylabels_log[plot_no], fontsize=20)
        else:
            while(i < ncurves):
                axes[plot_no].plot(z_grid, 10**param[i, :]*1e3, c='purple', **kwargs)
                i+=1
            axes[plot_no].set_ylabel(ylabels[plot_no], fontsize=20)

        axes[plot_no].set_xlabel('$z$', fontsize=20)


        plot_no+=1

def sample_allowed_parameter_curves(z_grid, nsamples, allowed_curves):

    sampled_curves = []

    for param in allowed_curves:
        ncurves = param.shape[0]
        #if(ncurves < nsamples):
        #    raise Exception("Requesting too many samples, try lowering nsamples")
        random_indexes = np.random.randint(0, ncurves, nsamples)
        param_df = pd.DataFrame(param, columns=z_grid)
        param_df = param_df.loc[random_indexes]
        sampled_curves.append(param_df.to_numpy())

    return sampled_curves