import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
import lbg_forecast.cosmology as cosmo
import emcee
import lbg_forecast.utils as utils

from uncertainties import unumpy

class GPModel(gpytorch.models.ExactGP):

        def __init__(self, train_x, train_y, lengthscale, scalefactor, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(lengthscale[0], lengthscale[1])), outputscale_prior=gpytorch.priors.SmoothedBoxPrior(scalefactor[0], scalefactor[1]))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MassFunctionPrior():

    def __init__(self, path, mean):

        self.path = path
        self.mean = mean

        #cell sizes
        self.dz = 0.001
        self.dlogm = 0.001

        #grids
        self.start_z = 0
        self.end_z = 7
        self.z_grid = np.linspace(self.start_z, self.end_z, int((self.end_z-self.start_z)/self.dz))

        self.start_logm = 7
        self.end_logm = 13
        self.logm_grid = np.linspace(self.start_logm, self.end_logm, int((self.end_logm-self.start_logm)/self.dlogm))

        self.dvdzgrid = self.dvdz_grid(self.z_grid, self.dz)

        self.param_names = ["$\mathrm{log}_{10}\phi^{*}$", "$\\alpha$", "$\mathrm{log}_{10}\mathcal{M}_{*}$"]
        state_dict_phi1 = torch.load(self.path+'/gp_models/phi1.pth', weights_only=True)
        state_dict_alpha1 = torch.load(self.path+'/gp_models/alpha1.pth', weights_only=True)
        state_dict_logm = torch.load(self.path+'/gp_models/logm.pth', weights_only=True)

        sorted_train_redshift_logphi1, sorted_train_logphi1, sorted_train_logphi1_errs = get_phi1_data(plotting=False)
        sorted_train_redshift_alpha1, sorted_train_alpha1, sorted_train_alpha1_errs = get_alpha1_data(plotting=False)
        sorted_train_redshift_logm, sorted_train_logm, sorted_train_logm_errs = get_logm_data(plotting=False)

        self.model_phi1 = create_gp_model([2.0, 999.0], [-999.0, 999.0], sorted_train_logphi1_errs, sorted_train_redshift_logphi1, sorted_train_logphi1)[0]
        self.model_phi1.load_state_dict(state_dict_phi1)
        self.model_phi1.eval()
        self.phi1_test_z = torch.linspace(0.0, 8.0, 100)

        self.model_alpha1 = create_gp_model([2.0, 999.0], [-999.0, 999.0], sorted_train_alpha1_errs, sorted_train_redshift_alpha1, sorted_train_alpha1)[0]
        self.model_alpha1.load_state_dict(state_dict_alpha1)
        self.model_alpha1.eval()
        self.alpha1_test_z = torch.linspace(0.0, 8.0, 100)

        self.model_logm = create_gp_model([2.0, 999.0], [-999.0, 999.0], sorted_train_logm_errs, sorted_train_redshift_logm, sorted_train_logm)[0]
        self.model_logm.load_state_dict(state_dict_logm)
        self.model_logm.eval()
        self.logm_test_z = torch.linspace(0.0, 8.0, 100)

        self.prior_phi1 = self.model_phi1(self.phi1_test_z)
        self.prior_alpha1 = self.model_alpha1(self.alpha1_test_z)
        self.prior_logm = self.model_logm(self.logm_test_z)

        self.priors = [self.prior_phi1, self.prior_alpha1, self.prior_logm]
        self.train_x = [sorted_train_redshift_logphi1, sorted_train_redshift_alpha1, sorted_train_redshift_logm]
        self.train_y = [sorted_train_logphi1, sorted_train_alpha1, sorted_train_logm]
        self.train_yerr = [sorted_train_logphi1_errs, sorted_train_alpha1_errs, sorted_train_logm_errs]
        self.test_x = [self.phi1_test_z, self.alpha1_test_z, self.logm_test_z]

    def mass_function(self, z, logm, sparams):
        """sparams=self.sample_prior(). logm can be array
        """

        logphi1, alpha1, logm_star = sparams
        logphi1 = np.interp(z, self.phi1_test_z, logphi1)
        alpha1 = np.interp(z, self.alpha1_test_z, alpha1)
        logm_star = np.interp(z, self.logm_test_z, logm_star)

        mfunc = self.schechter_function(logm, logphi1, logm_star, alpha1)
        return mfunc
    
    def lsst_number_density(self, sparams):
        i=0
        n_z=[]
        for z in self.z_grid:
            n_logm = self.mass_function(z, self.logm_grid, sparams)
            n_logm = np.trapz(n_logm, self.logm_grid)
            n_z.append(n_logm)
            i+=1

        return np.trapz(n_z*self.dvdzgrid, self.z_grid)/utils.FULL_SKY_DEG2
    
    def number_of_galaxies_in_volume(self, sparams, z1, z2, mmin):
        i=0
        n_z=[]
        indx=[]
        logms = self.logm_grid[np.where(self.logm_grid > mmin)[0]]
        for z in self.z_grid:
            if(z >= z1 and z <= z2):
                indx.append(i)
                n_logm = self.mass_function(z, logms, sparams)
                n_logm = np.trapz(n_logm, logms)
                n_z.append(n_logm)
                i+=1

        return np.trapz(n_z*self.dvdzgrid[indx], self.z_grid[indx])
    
    def surface_number_density_in_volume(self, sparams, z1, z2, mmin):
        return self.number_of_galaxies_in_volume(sparams, z1, z2, mmin)/utils.FULL_SKY_DEG2

    def n_tot(self, sparams):
        """mass function normalisation"""

        nz = self.n_z(sparams)
        return np.trapz(nz*self.dvdz_grid(self.z_grid, self.dz), self.z_grid)
    
    def n_z(self, sparams):
        n=[]
        for z in self.z_grid:
            phi_z_logm = self.mass_function(z, self.logm_grid, sparams)
            n.append(np.trapz(phi_z_logm, self.logm_grid))

        return np.array(n)
    
    def normalised_mass_function(self, z, logm, sparams, norm, dvdz):
        """norm=self.n_tot(sparams)"""
        return (self.mass_function(z, logm, sparams)/norm)*dvdz

    def volume_element(self, z, dz):
        return cosmo.get_cosmology().comoving_volume(z+dz).value - cosmo.get_cosmology().comoving_volume(z).value
    
    def volume_element_grid(self, grid, dz):
        vs = []
        for z in grid:
            vs.append(self.volume_element(z, dz))
        return np.array(vs)
    
    def dvdz_grid(self, grid, dz):
        dv = self.volume_element_grid(grid, dz)
        return dv/dz
    
    def dvdz(self, z, dz):
        dv = self.volume_element(z, dz)
        return dv/dz
    
    def logpdf(self, x, sparams, norm, prior_bounds):
        """log10(P(z, logm))"""

        z, logm = x

        if(z < prior_bounds[0] or z > prior_bounds[1]):
            return -np.inf
        
        if(logm < prior_bounds[2] or logm > prior_bounds[3]):
            return -np.inf
        
        if(logm < 8 and z > 1):
            return -np.inf

        p_z_logm = self.normalised_mass_function(z, logm, sparams, norm, np.interp(z, self.z_grid, self.dvdzgrid))

        if(p_z_logm<1e-100):
            return -np.inf
        else:
            log_p_z_logm = np.log10(p_z_logm)
            return log_p_z_logm
    
    def sample_logpdf(self, nsamples):

        burnin=1000
        nwalkers=100
        
        steps = int((burnin+nsamples)/nwalkers)


        #sampling
        ndim = 2
        pz = np.random.uniform(1.01, 1.02, (nwalkers, 1))
        plogm = np.random.uniform(9.9, 10.1, (nwalkers, 1))
        p0 = np.hstack((pz, plogm))
        prior_bounds=[0.0,7.0,7.0,13.0]

        if(self.mean):
            sparams = self.sample_prior_mean()
        else:
            sparams = self.sample_prior()

        print("Calculating Normalisation ... ")
        norm = self.n_tot(sparams)

        print("MCMC Sampling ... ")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logpdf, args=[sparams, norm, prior_bounds])
        state = sampler.run_mcmc(p0, 100)
        sampler.reset()

        sampler.run_mcmc(state, steps)

        print("MCMC Sampling Complete.")
        samples = sampler.get_chain(flat=True)
        redshift_samples = samples[:, 0]
        mass_samples = samples[:, 1]

        return redshift_samples[burnin:nsamples+burnin], mass_samples[burnin:nsamples+burnin], np.array(sparams)

    def schechter_function(self, logm, logphi, logm_star, alpha):
        return np.log(10)*(10**logphi)*10**((logm-logm_star)*(alpha+1))*np.exp(-10**(logm-logm_star))
    def double_schechter_function(self, logm, logphi1, logphi2, alpha1, alpha2, logm_star):
        return self.schechter_function(logm, logphi1, logm_star, alpha1) + self.schechter_function(logm, logphi2, logm_star, alpha2)
    
    def sample_prior(self):
        return[self.sample_phi1(), self.sample_alpha1(), self.sample_logm()]
    def sample_prior_mean(self):
        return[self.sample_phi1_mean(), self.sample_alpha1_mean(), self.sample_logm_mean()]
    
    def sample_phi1(self):
        return self.prior_phi1.sample().numpy()
    def sample_phi1_mean(self):
        return self.prior_phi1.mean.detach().numpy()
    
    def sample_alpha1(self):
        return self.prior_alpha1.sample().numpy()
    def sample_alpha1_mean(self):
        return self.prior_alpha1.mean.detach().numpy()
    
    def sample_logm(self):
        return self.prior_logm.sample().numpy()
    def sample_logm_mean(self):
        return self.prior_logm.mean.detach().numpy()
    
    def plot_prior_sample(self, prior_sample):

        f, ax = plt.subplots(5, 1, figsize=(7, 15), sharex=True)
        indx = 0
        for plot in ax:
            plot.plot(self.test_x[indx], prior_sample[indx])
            plot.set_ylabel(self.param_names[indx])
            plot.set_xlim(-0.2, 7.2)
            indx+=1

        plot.set_xlabel("redshift")
        plt.tight_layout()

    def plot_confidence(self):

        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(3, 1, figsize=(10, 20), sharex=True)

            indx = 0
            for plot in ax:
                
                prior = self.priors[indx]
                train_x = self.train_x[indx].numpy()
                train_y = self.train_y[indx].numpy()
                train_y_err = self.train_yerr[indx].numpy()
                test_x = self.test_x[indx]

                ms=10
                c='grey'
                capsize=7
                elinewidth=2
                capwidth=1
                est_error_colour = 'red'

                # Get upper and lower confidence bounds
                lower, upper = prior.confidence_region()
                # Plot training data

                if(indx==0):
                    san = np.where(train_x < 2.6)[0]
                    w = np.where((train_x < 3.5)&(train_x > 2.6))[0]
                    nc = np.where(train_x > 3.5)[0]
                    nc8 = np.where(train_x > 7.0)[0]

                    plot.errorbar(train_x[san], train_y[san], train_y_err[san], fmt='d', color=c, capsize=capsize, ms=ms, label='Santini et al. (2021)', elinewidth=elinewidth, capthick=capwidth, ecolor='k')
                    plot.errorbar(train_x[w], train_y[w], train_y_err[w], fmt='o', color=c, capsize=capsize, ms=ms, label='Weaver et al. (2023)', elinewidth=elinewidth, capthick=capwidth, ecolor='k')
                    plot.errorbar(train_x[nc], train_y[nc], train_y_err[nc], fmt='v', color=c, capsize=capsize, ms=ms, label='Navarro-Carrera et al. (2024)', elinewidth=elinewidth, capthick=capwidth, ecolor='k')
                    plot.errorbar(train_x[nc8], train_y[nc8], train_y_err[nc8], fmt='v', color=c, capsize=capsize, ms=ms, elinewidth=elinewidth, capthick=capwidth, ecolor='k')


                    # Plot predictive means as blue line
                    plot.plot(test_x.numpy(), prior.mean, color='purple', lw=5, label='Gaussian Process Mean')
                    # Shade between the lower and upper confidence bounds
                    plot.fill_between(test_x.numpy(), lower, upper, alpha=0.25, color='purple', label='Gaussian Process 2$\sigma$ Confidence', lw=0)

                    plot.legend(fontsize=14)

                else:
                    plot.errorbar(train_x[san], train_y[san], train_y_err[san], fmt='d', color=c, capsize=capsize, ms=ms, label='Santini et al. (2023)', elinewidth=elinewidth, capthick=capwidth, ecolor='k')
                    if(indx == 1):
                        plot.errorbar(train_x[w], train_y[w], train_y_err[w], fmt='o', color=est_error_colour, capsize=capsize, ms=ms, label='Weaver et al. (2023)', elinewidth=elinewidth, capthick=capwidth, ecolor=est_error_colour)
                    else:
                        plot.errorbar(train_x[w], train_y[w], train_y_err[w], fmt='o', color=c, capsize=capsize, ms=ms, label='Weaver et al. (2023)', elinewidth=elinewidth, capthick=capwidth, ecolor='k') 
                    plot.errorbar(train_x[nc], train_y[nc], train_y_err[nc], fmt='v', color=c, capsize=capsize, ms=ms, label='Navarro-Carrera et al. (2023)', elinewidth=elinewidth, capthick=capwidth, ecolor='k')
                    if(indx == 2):
                        plot.errorbar(train_x[nc8], train_y[nc8], train_y_err[nc8], fmt='v', color=est_error_colour, capsize=capsize, ms=ms, elinewidth=elinewidth, capthick=capwidth, ecolor=est_error_colour)
                    else:
                        plot.errorbar(train_x[nc8], train_y[nc8], train_y_err[nc8], fmt='v', color=c, capsize=capsize, ms=ms, elinewidth=elinewidth, capthick=capwidth, ecolor='k')

                    # Plot predictive means as blue line
                    plot.plot(test_x.numpy(), prior.mean, color='purple', lw=5)
                    # Shade between the lower and upper confidence bounds
                    plot.fill_between(test_x.numpy(), lower, upper, alpha=0.25, color='purple', lw=0)

                #ax.legend(['Observed Data', 'Mean', 'Confidence'])
                plot.set_ylabel(self.param_names[indx], fontsize=32)


                if(indx==1 or indx==3):
                    plot.set_xlim(-0.2, 8.2)
                    plot.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
                else:
                    plot.set_xlim(-0.2, 8.2)
                    plot.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])

                plot.tick_params('x', labelsize=20, direction='in', width=2, size=7, top=True)
                plot.tick_params('y', labelsize=20, direction='in', width=2, size=7, right=True)
                plot.minorticks_on()
                plot.tick_params(axis='x', which='minor', direction='in', size=5, top=True)
                plot.tick_params(axis='y', which='minor', direction='in', size=5, right=True)

                plot.grid(visible=True, alpha=0.2)

                indx+=1

            plot.set_xlabel("Redshift", fontsize=32)
            plt.tight_layout()

def create_gp_model(lengthscale, scalefactor, errors, train_x, train_y):

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.square(errors), learn_additional_noise=False, noise_constraint=gpytorch.constraints.GreaterThan(0.0))
    model = GPModel(train_x, train_y, lengthscale, scalefactor, likelihood)

    return model, likelihood

def gp_training_loop(model, likelihood, train_x, train_y, training_iter, lr=1e-4):

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #i + 1, training_iter, loss.item(),
        print(loss.item())
            #model.likelihood.noise.item()
        #))
        optimizer.step()

    return model, likelihood

def gp_evaluate_model(model, likelihood, test_x):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    f_preds = model(test_x)

    return f_preds

def gp_plot_confidence(f_preds, test_x, train_x, train_y, trainyerr, labely):

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7))

        # Get upper and lower confidence bounds
        lower, upper = f_preds.confidence_region()
        # Plot training data as black stars
        ax.errorbar(train_x.numpy(), train_y.numpy(), yerr=trainyerr.numpy(), fmt='ko', capsize=2)
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), f_preds.mean, 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower, upper, alpha=0.5)
        #ax.legend(['Observed Data', 'Mean', 'Confidence'])

        ax.set_xlabel("redshift")
        ax.set_ylabel(labely)

def gp_plot_realisations(f_preds, test_x, train_x, train_y, trainyerr, labely):

    with torch.no_grad():

        f, ax = plt.subplots(1, 1, figsize=(10, 7))

        nsamples = 1000
        for sample in range(nsamples):
            f_sample = f_preds.sample()
            #if((f_sample > 0.0).all()):
            ax.plot(test_x, f_sample, c='purple', alpha=0.1)
        ax.plot(test_x, f_preds.mean, zorder=1000, ls='-', c='k')
        ax.errorbar(train_x.numpy(), train_y.numpy(), yerr=trainyerr.numpy(), fmt='ko', capsize=2)
        
        ax.set_xlabel("redshift")
        ax.set_ylabel(labely)

def gp(train_x, train_y, train_y_errs, test_x, lengthscale, scalefactor, lr, training_iter, ylabel, name):

    #initialise
    model, likelihood = create_gp_model(lengthscale, scalefactor, train_y_errs, train_x, train_y)

    #print params
    for param_name, param in model.named_parameters():
        print(f'Parameter name: {param_name:42} value = {param.item()}')

    #train model
    trained_model, trained_likelihood =  gp_training_loop(model, likelihood, train_x, train_y, training_iter=training_iter, lr=lr)

    #evaluate model
    f_preds = gp_evaluate_model(trained_model, trained_likelihood, test_x)

    #print parameter values
    for param_name, param in model.named_parameters():
        print(f'Parameter name: {param_name:42} value = {param.item()}')

    #plotting
    gp_plot_confidence(f_preds, test_x, train_x, train_y, train_y_errs, ylabel)
    gp_plot_realisations(f_preds, test_x, train_x, train_y, train_y_errs, ylabel)

    #save
    torch.save(trained_model.state_dict(), 'gp_models/'+name+'.pth')

def get_phi1_data(plotting=False):

    #nc 2024
    nc_low_mass_norm_val = np.array([1.65e-4, 9.6e-5, 6.3e-5, 1.3e-5, 2.8e-6])
    nc_low_mass_norm_errl = np.array([0.04e-4, 0.9e-5, 1.0e-5, 1.0e-5, 2.1e-6])
    nc_low_mass_norm_errh = np.array([0.04e-4, 0.9e-5, 1.0e-5, 1.0e-5, 2.1e-6])
    nc_low_mass_norm_errs = (nc_low_mass_norm_errl + nc_low_mass_norm_errh)/2
    nc_redshift = np.array([4.0, 5.0, 6.0, 7.0, 8.0])

    #weaver 2023
    w_low_mass_norm_val = np.array([0.24e-3, 0.21e-3])
    w_low_mass_norm_errl = np.array([0.02e-3, 0.03e-3])
    w_low_mass_norm_errh = np.array([0.03e-3, 0.03e-3])
    w_low_mass_norm_errs = (w_low_mass_norm_errl + w_low_mass_norm_errh)/2
    w_redshift = np.array([2.75, 3.25])

    #santini 2022
    log_san_low_mass_norm_val = np.array([-2.96, -3.20, -3.45, -3.50, -3.76])
    san_low_mass_norm_log_errl = np.array([0.03, 0.03, 0.04, 0.04, 0.07])
    san_low_mass_norm_log_errh = np.array([0.03, 0.03, 0.04, 0.04, 0.07])
    san_low_mass_norm_log_errs = (san_low_mass_norm_log_errl + san_low_mass_norm_log_errh)/2

    san_redshift_lower_bin_edge = np.array([0.25, 0.75, 1.25, 1.75, 2.25])
    san_redshift_upper_bin_edge = np.array([0.75, 1.25, 1.75, 2.25, 2.75])
    san_redshift_midpoint = (san_redshift_lower_bin_edge + san_redshift_upper_bin_edge)/2

    #stefanon 2021
    stef_low_mass_norm_val = np.array([-4.09, -4.14, -4.69])
    stef_low_mass_norm_errl = np.array([0.12, 0.23, 0.72])
    stef_low_mass_norm_errh = np.array([0.17, 0.19, 0.40])
    stef_low_mass_norm_errs = (stef_low_mass_norm_errl + stef_low_mass_norm_errh)/2
    stef_redshift = np.array([6, 7, 8])

    #song 2016
    song_low_mass_norm_val = np.array([26.15e-5, 5.43e-5])
    song_low_mass_norm_errl = np.array([14.08e-5, 4.04e-5])
    song_low_mass_norm_errh = np.array([24.11e-5, 8.22e-5])
    song_low_mass_norm_errs = (song_low_mass_norm_errl + song_low_mass_norm_errh)/2

    song_redshift = np.array([4, 5])

    #mortlock 2015
    mo_low_mass_norm_val = np.array([-2.54, -2.71, -3.21, -3.74, -3.78, -4.03])
    mo_low_mass_norm_errl = np.array([0.13, 0.10, 0.06, 0.09, 0.14, 0.16])
    mo_low_mass_norm_errh = np.array([0.13, 0.10, 0.06, 0.09, 0.14, 0.16])
    mo_low_mass_norm_errs = (mo_low_mass_norm_errl + mo_low_mass_norm_errh)/2

    mo_redshift_lower_bin_edge = np.array([0.3, 0.5, 1.0, 1.5, 2.0, 2.5])
    mo_redshift_upper_bin_edge = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    mo_redshift_midpoint = (mo_redshift_lower_bin_edge + mo_redshift_upper_bin_edge)/2

    #caputi 2015
    caputi_low_mass_norm_val = np.array([1.23e-4, 6.14e-5])
    caputi_low_mass_norm_errl = np.array([0.26e-4, 1.52e-5])
    caputi_low_mass_norm_errh = np.array([0.95e-4, 3.55e-5])
    caputi_low_mass_norm_errs = (caputi_low_mass_norm_errl + caputi_low_mass_norm_errh)/2
    caputi_redshift = np.array([3.5, 4.5])

    #weibel 2024
    log_weibel_low_mass_norm_val = np.array([-4.52, -4.07, -4.26])
    weibel_low_mass_norm_log_errl = np.array([0.14, 0.14, 0.36])
    weibel_low_mass_norm_log_errh = np.array([0.13, 0.13, 0.36])
    weibel_low_mass_norm_log_errs = (weibel_low_mass_norm_log_errl + weibel_low_mass_norm_log_errh)/2

    weibel_redshift = np.array([4, 5, 6])

    #mcleod 2021
    log_mcleod_low_mass_norm_val = np.array([-2.99, -3.01, -3.15, -3.41, -3.54, -4.05])
    mcleod_low_mass_norm_log_errl = np.array([0.03, 0.05, 0.04, 0.06, 0.08, 0.13])
    mcleod_low_mass_norm_log_errh = np.array([0.03, 0.04, 0.04, 0.05, 0.06, 0.10])
    mcleod_low_mass_norm_log_errs = (mcleod_low_mass_norm_log_errl + mcleod_low_mass_norm_log_errh)/2

    mcleod_redshift_lower_bin_edge = np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    mcleod_redshift_upper_bin_edge = np.array([0.75, 1.25, 1.75, 2.25, 2.75, 3.75])
    mcleod_redshift_midpoint = (mcleod_redshift_lower_bin_edge + mcleod_redshift_upper_bin_edge)/2

    #grazian 2015
    log_gra_low_mass_norm_val = np.array([-3.94, -4.18, -4.16, -5.24])
    gra_low_mass_norm_log_errl = np.array([0.16, 0.29, 0.47, 2.02])
    gra_low_mass_norm_log_errh = np.array([0.16, 0.29, 0.47, 2.02])
    gra_low_mass_norm_log_errs = (gra_low_mass_norm_log_errl + gra_low_mass_norm_log_errh)/2

    gra_redshift_lower_bin_edge = np.array([3.5, 4.5, 5.5, 6.5])
    gra_redshift_upper_bin_edge = np.array([4.5, 5.5, 6.7, 7.5])
    gra_redshift_midpoint = (gra_redshift_lower_bin_edge + gra_redshift_upper_bin_edge)/2

    #kiku 2020
    kiku_low_mass_norm_val = np.array([45.7e-5, 60.3e-5])
    kiku_low_mass_norm_log_errl = np.array([25.7e-5, 46.8e-5])
    kiku_low_mass_norm_log_errh = np.array([33.7e-5, 37.5e-5])
    kiku_low_mass_norm_log_errs = (kiku_low_mass_norm_log_errl + kiku_low_mass_norm_log_errh)/2

    kiku_redshift = np.array([8, 9])

    #nc error prop
    log_nc_val, log_nc_err = log_error_prop(nc_low_mass_norm_val, nc_low_mass_norm_errs)

    #song error prop
    log_song_val, log_song_err = log_error_prop(song_low_mass_norm_val, song_low_mass_norm_errs)
    
    #caputi error prop
    log_caputi_val, log_caputi_err = log_error_prop(caputi_low_mass_norm_val, caputi_low_mass_norm_errs)

    #w error prop
    log_w_val, log_w_err = log_error_prop(w_low_mass_norm_val, w_low_mass_norm_errs)

    #kiku error prop
    log_kiku_val, log_kiku_err = log_error_prop(kiku_low_mass_norm_val, kiku_low_mass_norm_log_errs)

    #train_logphi1 = torch.from_numpy(np.concatenate((log_nc_val, log_san_low_mass_norm_val, stef_low_mass_norm_val, log_song_val, mo_low_mass_norm_val, log_caputi_val, log_weibel_low_mass_norm_val, log_mcleod_low_mass_norm_val, log_gra_low_mass_norm_val)))
    #train_logphi1_errs = torch.from_numpy(np.concatenate((log_nc_err, san_low_mass_norm_log_errs, stef_low_mass_norm_errs, log_song_err, mo_low_mass_norm_errs, log_caputi_err, weibel_low_mass_norm_log_errs, mcleod_low_mass_norm_log_errs, gra_low_mass_norm_log_errs)))
    #train_redshift = torch.from_numpy(np.concatenate((nc_redshift, san_redshift_midpoint, stef_redshift, song_redshift, mo_redshift_midpoint, caputi_redshift, weibel_redshift, mcleod_redshift_midpoint, gra_redshift_midpoint)))

    #train_logphi1 = torch.from_numpy(np.concatenate((log_nc_val, log_san_low_mass_norm_val, mo_low_mass_norm_val, log_mcleod_low_mass_norm_val)))
    #train_logphi1_errs = torch.from_numpy(np.concatenate((log_nc_err, san_low_mass_norm_log_errs, mo_low_mass_norm_errs, mcleod_low_mass_norm_log_errs)))
    #train_redshift = torch.from_numpy(np.concatenate((nc_redshift, san_redshift_midpoint, mo_redshift_midpoint, mcleod_redshift_midpoint)))

    train_logphi1 = torch.from_numpy(np.concatenate((log_nc_val, log_san_low_mass_norm_val, log_w_val)))
    train_logphi1_errs = torch.from_numpy(np.concatenate((log_nc_err, san_low_mass_norm_log_errs, log_w_err)))
    train_redshift = torch.from_numpy(np.concatenate((nc_redshift, san_redshift_midpoint, w_redshift)))

    sorted_redshift_inds = train_redshift.argsort()[:]
    sorted_train_logphi1 = train_logphi1[sorted_redshift_inds]
    sorted_train_logphi1_errs = train_logphi1_errs[sorted_redshift_inds]
    sorted_train_redshift = train_redshift[sorted_redshift_inds]

    if(plotting):

        flogphi, ax1 = plt.subplots(1, 1, figsize=(10, 7))
        ax1.errorbar(sorted_train_redshift, sorted_train_logphi1, sorted_train_logphi1_errs, fmt='ko')
        ax1.set_ylabel("Low Mass Normalisation $log_{10}(\phi_{1})$")
        ax1.set_xlabel("redshift")
    
    return sorted_train_redshift, sorted_train_logphi1, sorted_train_logphi1_errs


def get_alpha1_data(plotting=False):

    #nc 2024
    nc_alpha_val = np.array([-1.61, -1.69, -1.88, -1.98, -1.93])
    nc_alpha_errl = np.array([0.06, 0.07, 0.09, 0.14, 0.22])
    nc_alpha_errh = np.array([0.06, 0.07, 0.09, 0.14, 0.22])
    nc_alpha_errs = (nc_alpha_errl + nc_alpha_errh)/2
    nc_redshift = np.array([4.0, 5.0, 6.0, 7.0, 8.0])

    #weaver 2023
    w_alpha_val = np.array([-1.46, -1.46])
    w_alpha_errl = np.array([0.06, 0.06])
    w_alpha_errh = np.array([0.05, 0.05])
    w_alpha_errs = (w_alpha_errl + w_alpha_errh)/2
    w_redshift = np.array([2.75, 3.25])

    #santini 2022
    san_alpha_val = np.array([-1.36, -1.41, -1.50, -1.45, -1.61])
    san_alpha_errl = np.array([0.01, 0.01, 0.01, 0.01, 0.03])
    san_alpha_errh = np.array([0.01, 0.01, 0.01, 0.01, 0.03])
    san_alpha_errs = (san_alpha_errl + san_alpha_errh)/2

    san_redshift_lower_bin_edge = np.array([0.25, 0.75, 1.25, 1.75, 2.25])
    san_redshift_upper_bin_edge = np.array([0.75, 1.25, 1.75, 2.25, 2.75])
    san_redshift_midpoint = (san_redshift_lower_bin_edge + san_redshift_upper_bin_edge)/2

    #stefanon 2021
    stef_alpha_val = np.array([-1.88, -1.73, -1.82])
    stef_alpha_errl = np.array([0.03, 0.08, 0.21])
    stef_alpha_errh = np.array([0.06, 0.08, 0.20])
    stef_alpha_errs = (stef_alpha_errl + stef_alpha_errh)/2
    stef_redshift = np.array([6, 7, 8])

    #song 2015
    song_alpha_val = np.array([-1.54, -1.69])
    song_alpha_errl = np.array([0.07, 0.08, 0.10])
    song_alpha_errh = np.array([0.08, 0.09, 0.10])
    song_alpha_errs = (song_alpha_errl + song_alpha_errh)/2
    song_redshift = np.array([4, 5])

    #mortlock 2015
    mo_alpha_val = np.array([-1.59, -1.42, -1.31, -1.51, -1.56, -1.69])
    mo_alpha_errl = np.array([0.08, 0.06, 0.03, 0.03, 0.06, 0.06])
    mo_alpha_errh = np.array([0.08, 0.06, 0.03, 0.03, 0.06, 0.06])
    mo_alpha_errs = (mo_alpha_errl + mo_alpha_errh)/2

    mo_redshift_lower_bin_edge = np.array([0.3, 0.5, 1.0, 1.5, 2.0, 2.5])
    mo_redshift_upper_bin_edge = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    mo_redshift_midpoint = (mo_redshift_lower_bin_edge + mo_redshift_upper_bin_edge)/2

    #caputi 2015
    caputi_alpha_val = np.array([-1.58, -1.58])
    caputi_alpha_errl = np.array([0.06, 0.16])
    caputi_alpha_errh = np.array([0.06, 0.12])
    caputi_alpha_errs = (caputi_alpha_errl + caputi_alpha_errh)/2
    caputi_redshift = np.array([3.5, 4.5])

    #weibel 2024
    weibel_alpha_val = np.array([-1.79, -1.86, -1.95])
    weibel_alpha_errl = np.array([0.01, 0.03, 0.06])
    weibel_alpha_errh = np.array([0.01, 0.03, 0.08])
    weibel_alpha_errs = (weibel_alpha_errl + weibel_alpha_errh)/2
    weibel_redshift = np.array([4, 5, 6])

    #mcleod 2021
    mcleod_alpha_val = np.array([-1.37, -1.37, -1.36, -1.43, -1.51, -1.79])
    mcleod_alpha_errl = np.array([0.01, 0.02, 0.03, 0.03, 0.04, 0.05])
    mcleod_alpha_errh = np.array([0.01, 0.02, 0.03, 0.03, 0.04, 0.05])
    mcleod_alpha_errs = (mcleod_alpha_errl + mcleod_alpha_errh)/2

    mcleod_redshift_lower_bin_edge = np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    mcleod_redshift_upper_bin_edge = np.array([0.75, 1.25, 1.75, 2.25, 2.75, 3.75])
    mcleod_redshift_midpoint = (mcleod_redshift_lower_bin_edge + mcleod_redshift_upper_bin_edge)/2

    #grazian 2015
    gra_alpha_val = np.array([-1.63, -1.63, -1.55, -1.88])
    gra_alpha_errl = np.array([0.05, 0.09, 0.19, 0.36])
    gra_alpha_errh = np.array([0.05, 0.09, 0.19, 0.36])
    gra_alpha_errs = (gra_alpha_errl + gra_alpha_errh)/2

    gra_redshift_lower_bin_edge = np.array([3.5, 4.5, 5.5, 6.5])
    gra_redshift_upper_bin_edge = np.array([4.5, 5.5, 6.7, 7.5])
    gra_redshift_midpoint = (gra_redshift_lower_bin_edge + gra_redshift_upper_bin_edge)/2

    kiku_alpha_val = np.array([-1.52, -1.55])
    kiku_alpha_errl = np.array([0.26, 0.30])
    kiku_alpha_errh = np.array([0.27, 0.29])
    kiku_alpha_errs = (kiku_alpha_errl + kiku_alpha_errh)/2
    kiku_redshift = np.array([8, 9])

    #train_alpha1 = torch.from_numpy(np.concatenate((nc_alpha_val, san_alpha_val, stef_alpha_val, song_alpha_val, mo_alpha_val, caputi_alpha_val, weibel_alpha_val, mcleod_alpha_val, gra_alpha_val)))
    #train_alpha1_errs = torch.from_numpy(np.concatenate((nc_alpha_errs, san_alpha_errs, stef_alpha_errs, song_alpha_errs, mo_alpha_errs, caputi_alpha_errs, weibel_alpha_errs, mcleod_alpha_errs, gra_alpha_errs)))
    #train_redshift = torch.from_numpy(np.concatenate((nc_redshift, san_redshift_midpoint, stef_redshift, song_redshift, mo_redshift_midpoint, caputi_redshift, weibel_redshift, mcleod_redshift_midpoint, gra_redshift_midpoint)))

    #train_alpha1 = torch.from_numpy(np.concatenate((nc_alpha_val, san_alpha_val, mo_alpha_val, mcleod_alpha_val)))
    #train_alpha1_errs = torch.from_numpy(np.concatenate((nc_alpha_errs, san_alpha_errs, mo_alpha_errs, mcleod_alpha_errs)))
    #train_redshift = torch.from_numpy(np.concatenate((nc_redshift, san_redshift_midpoint, mo_redshift_midpoint, mcleod_redshift_midpoint)))

    train_alpha1 = torch.from_numpy(np.concatenate((nc_alpha_val, san_alpha_val, w_alpha_val)))
    train_alpha1_errs = torch.from_numpy(np.concatenate((nc_alpha_errs, san_alpha_errs, w_alpha_errs)))
    train_redshift = torch.from_numpy(np.concatenate((nc_redshift, san_redshift_midpoint, w_redshift)))

    sorted_redshift_inds = train_redshift.argsort()[:]
    sorted_train_alpha1 = train_alpha1[sorted_redshift_inds]
    sorted_train_alpha1_errs = train_alpha1_errs[sorted_redshift_inds]
    sorted_train_redshift = train_redshift[sorted_redshift_inds]

    if(plotting):

        plt.errorbar(sorted_train_redshift, sorted_train_alpha1, sorted_train_alpha1_errs, fmt='ko')
        plt.ylabel("Low Mass Slope $\\alpha_{1}$")
        plt.xlabel("redshift")

    return sorted_train_redshift, sorted_train_alpha1, sorted_train_alpha1_errs

def get_logm_data(plotting=False):

    #nc 2024 (c-imf)
    nc_logm_val = np.array([10.48, 10.45, 10.33, 10.68, 10.70])
    nc_logm_errl = np.array([0.15, 0.27, 0.36, 0.79, 0.79])
    nc_logm_errh = np.array([0.15, 0.27, 0.36, 0.79, 0.79])
    nc_logm_errs = (nc_logm_errl + nc_logm_errh)/2
    nc_redshift = np.array([4.0, 5.0, 6.0, 7.0, 8.0])

    #weaver 2023 (c-imf)
    w_logm_val = np.array([10.97, 10.83])
    w_logm_errl = np.array([0.07, 0.09])
    w_logm_errh = np.array([0.06, 0.11])
    w_logm_errs = (w_logm_errl + w_logm_errh)/2
    w_redshift = np.array([2.75, 3.25])

    #santini 2022 (c-imf)
    san_logm_val = np.array([10.98, 11.08, 11.11, 11.10, 11.05])
    san_logm_errl = np.array([0.03, 0.03, 0.03, 0.03, 0.06])
    san_logm_errh = np.array([0.03, 0.03, 0.03, 0.03, 0.06])
    san_logm_errs = (san_logm_errl + san_logm_errh)/2

    san_redshift_lower_bin_edge = np.array([0.25, 0.75, 1.25, 1.75, 2.25])
    san_redshift_upper_bin_edge = np.array([0.75, 1.25, 1.75, 2.25, 2.75])
    san_redshift_midpoint = (san_redshift_lower_bin_edge + san_redshift_upper_bin_edge)/2

    #stefanon 2021 (salpeter)
    stef_logm_val = np.array([10.24, 10.04, 9.98])
    stef_logm_errl = np.array([0.11, 0.13, 0.24])
    stef_logm_errh = np.array([0.08, 0.15, 0.44])
    stef_logm_errs = (stef_logm_errl + stef_logm_errh)/2
    stef_redshift = np.array([6, 7, 8])

    #song 2016 (salpeter)
    song_logm_val = np.array([10.49, 10.95])
    song_logm_errl = np.array([0.26, 0.38])
    song_logm_errh = np.array([0.33, 0.56])
    song_logm_errs = (song_logm_errl + song_logm_errh)/2
    song_redshift = np.array([4, 5])

    #mortlock 2015 (c-imf)
    mo_logm_val = np.array([10.90, 10.90, 11.04, 11.15, 11.02, 11.04])
    mo_logm_errl = np.array([0.13, 0.11, 0.04, 0.06, 0.10, 0.11])
    mo_logm_errh = np.array([0.13, 0.11, 0.04, 0.06, 0.10, 0.11])
    mo_logm_errs = (mo_logm_errl + mo_logm_errh)/2

    mo_redshift_lower_bin_edge = np.array([0.3, 0.5, 1.0, 1.5, 2.0, 2.5])
    mo_redshift_upper_bin_edge = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    mo_redshift_midpoint = (mo_redshift_lower_bin_edge + mo_redshift_upper_bin_edge)/2

    #caputi 2015 (salpeter)
    caputi_logm_val = np.array([1.79e11, 2.0e11])
    caputi_logm_errl = np.array([0.21e11, 0.59e11])
    caputi_logm_errh = np.array([0.16e11, 1.16e11])
    caputi_logm_errs = (caputi_logm_errl + caputi_logm_errh)/2
    caputi_redshift = np.array([3.5, 4.5])

    #weibel 2024 (kroupa+)
    weibel_alpha_val = np.array([11.01, 10.26, 10.01])
    weibel_logm_errl = np.array([0.14, 0.14, 0.36])
    weibel_logm_errh = np.array([0.14, 0.11, 0.28])
    weibel_alpha_errs = (weibel_logm_errl + weibel_logm_errh)/2
    weibel_redshift = np.array([4, 5, 6])

    #mcleod 2021 (c-imf)
    mcleod_logm_val = np.array([10.96, 10.86, 10.88, 10.90, 10.91, 10.97])
    mcleod_logm_errl = np.array([0.03, 0.03, 0.02, 0.03, 0.04, 0.07])
    mcleod_logm_errh = np.array([0.03, 0.03, 0.02, 0.03, 0.04, 0.07])
    mcleod_logm_errs = (mcleod_logm_errl + mcleod_logm_errh)/2

    mcleod_redshift_lower_bin_edge = np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    mcleod_redshift_upper_bin_edge = np.array([0.75, 1.25, 1.75, 2.25, 2.75, 3.75])
    mcleod_redshift_midpoint = (mcleod_redshift_lower_bin_edge + mcleod_redshift_upper_bin_edge)/2
    
    #grazian 2015 (salpeter)
    gra_logm_val = np.array([10.96, 10.78, 10.49, 10.69])
    gra_logm_errl = np.array([0.13, 0.23, 0.32, 1.58])
    gra_logm_errh = np.array([0.13, 0.23, 0.32, 1.58])
    gra_logm_errs = (gra_logm_errl + gra_logm_errh)/2

    gra_redshift_lower_bin_edge = np.array([3.5, 4.5, 5.5, 6.5])
    gra_redshift_upper_bin_edge = np.array([4.5, 5.5, 6.7, 7.5])
    gra_redshift_midpoint = (gra_redshift_lower_bin_edge + gra_redshift_upper_bin_edge)/2

    kiku_logm_val = np.array([8.93, 9.04])
    kiku_logm_errl = np.array([0.15, 0.18])
    kiku_logm_errh = np.array([0.23, 0.47])
    kiku_logm_errs = (kiku_logm_errl + kiku_logm_errh)/2
    kiku_redshift = np.array([8, 9])

    #caputi error prop
    caputi_val, caputi_err = log_error_prop(caputi_logm_val, caputi_logm_errs)
 
    #train_logm = torch.from_numpy(np.concatenate((nc_logm_val, san_logm_val, stef_logm_val, song_logm_val, mo_logm_val, caputi_val, weibel_alpha_val, mcleod_logm_val, gra_logm_val)))
    #train_logm_errs = torch.from_numpy(np.concatenate((nc_logm_errs, san_logm_errs, stef_logm_errs, song_logm_errs, mo_logm_errs, caputi_err, weibel_alpha_errs, mcleod_logm_errs, gra_logm_errs)))
    #train_redshift = torch.from_numpy(np.concatenate((nc_redshift, san_redshift_midpoint, stef_redshift, song_redshift, mo_redshift_midpoint, caputi_redshift, weibel_redshift, mcleod_redshift_midpoint, gra_redshift_midpoint)))

    #train_logm = torch.from_numpy(np.concatenate((nc_logm_val, san_logm_val, mo_logm_val, mcleod_logm_val)))
    #train_logm_errs = torch.from_numpy(np.concatenate((nc_logm_errs, san_logm_errs, mo_logm_errs, mcleod_logm_errs)))
    #train_redshift = torch.from_numpy(np.concatenate((nc_redshift, san_redshift_midpoint, mo_redshift_midpoint, mcleod_redshift_midpoint)))

    train_logm = torch.from_numpy(np.concatenate((nc_logm_val, san_logm_val, w_logm_val)))
    train_logm_errs = torch.from_numpy(np.concatenate((nc_logm_errs, san_logm_errs, w_logm_errs)))
    train_redshift = torch.from_numpy(np.concatenate((nc_redshift, san_redshift_midpoint, w_redshift)))

    sorted_redshift_inds = train_redshift.argsort()[:]
    sorted_train_logm = train_logm[sorted_redshift_inds]
    sorted_train_logm_errs = train_logm_errs[sorted_redshift_inds]
    sorted_train_redshift = train_redshift[sorted_redshift_inds]

    if(plotting):
        plt.errorbar(sorted_train_redshift, sorted_train_logm, sorted_train_logm_errs, fmt='ko')
        plt.ylabel("Crossover $\mathrm{LogM}_{*}$")
        plt.xlabel("redshift")

    return sorted_train_redshift, sorted_train_logm, sorted_train_logm_errs

def get_phi2_data(plotting=False):

    #weaver
    weaver_high_mass_norm_val = np.array([1.09, 0.83, 0.66, 0.34, 0.64, 0.27])
    weaver_high_mass_norm_errl = np.array([0.54, 0.43, 0.42, 0.25, 0.16, 0.12])
    weaver_high_mass_norm_errh = np.array([0.5, 0.37, 0.34, 0.30, 0.13, 0.12])
    weaver_high_mass_norm_errs = (weaver_high_mass_norm_errl+weaver_high_mass_norm_errh)/2#np.maximum(weaver_high_mass_norm_errl, weaver_high_mass_norm_errh)

    weaver_redshift_lower_bin_edge_lowz = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0])
    weaver_redshift_upper_bin_edge_lowz = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5])
    weaver_redshift_midpoint_lowz = (weaver_redshift_lower_bin_edge_lowz + weaver_redshift_upper_bin_edge_lowz)/2

    #cont
    log_cont_high_mass_norm_val = np.array([-2.44, -3.08, -4.14])
    cont_high_mass_norm_log_errl = np.array([0.02, 0.03, 0.11])
    cont_high_mass_norm_log_errh = np.array([0.02, 0.02, 0.10])
    cont_high_mass_norm_log_errs = (cont_high_mass_norm_log_errl+cont_high_mass_norm_log_errh)/2#np.maximum(cont_high_mass_norm_log_errl, cont_high_mass_norm_log_errh)

    cont_redshift_anchor_points = np.array([0.2, 1.6, 3.0])

    #mcleod
    log_mcleod_high_mass_norm_val = np.array([-2.37, -2.63, -2.67, -2.83, -3.05, -3.28, -4.08])
    mcleod_high_mass_norm_log_errl = np.array([0.04, 0.05, 0.07, 0.04, 0.07, 0.10, 0.33])
    mcleod_high_mass_norm_log_errh = np.array([0.03, 0.05, 0.06, 0.04, 0.06, 0.08, 0.18])
    mcleod_high_mass_norm_log_errs = (mcleod_high_mass_norm_log_errl+mcleod_high_mass_norm_log_errh)/2#np.maximum(mcleod_high_mass_norm_log_errl, mcleod_high_mass_norm_log_errh)

    mcleod_redshift_lower_bin_edge = np.array([0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    mcleod_redshift_upper_bin_edge = np.array([0.06, 0.75, 1.25, 1.75, 2.25, 2.75, 3.75001])
    mcleod_redshift_midpoint = (mcleod_redshift_lower_bin_edge + mcleod_redshift_upper_bin_edge)/2

    weaver_arr = unumpy.uarray(weaver_high_mass_norm_val, weaver_high_mass_norm_errs)
    log_weaver_arr = unumpy.log10((10**-3)*weaver_arr)

    log_weaver_val = unumpy.nominal_values(log_weaver_arr)
    log_weaver_err = unumpy.std_devs(log_weaver_arr)

    train_logphi2 = torch.from_numpy(np.concatenate((log_weaver_val, log_cont_high_mass_norm_val, log_mcleod_high_mass_norm_val)))
    train_logphi2_errs = torch.from_numpy(np.concatenate((log_weaver_err, cont_high_mass_norm_log_errs, mcleod_high_mass_norm_log_errs)))
    train_redshift = torch.from_numpy(np.concatenate((weaver_redshift_midpoint_lowz, cont_redshift_anchor_points, mcleod_redshift_midpoint)))

    sorted_redshift_inds = train_redshift.argsort()[:]
    sorted_train_logphi2 = train_logphi2[sorted_redshift_inds]
    sorted_train_logphi2_errs = train_logphi2_errs[sorted_redshift_inds]
    sorted_train_redshift = train_redshift[sorted_redshift_inds]

    if(plotting):

        flogphi, ax2 = plt.subplots(1, 1, figsize=(10, 7))
        ax2.errorbar(sorted_train_redshift, sorted_train_logphi2, yerr=sorted_train_logphi2_errs, fmt='ko')
        ax2.set_ylabel("High Mass Normalisation $log_{10}(\phi_{2})$")
        ax2.set_xlabel("redshift")

    return sorted_train_redshift, sorted_train_logphi2, sorted_train_logphi2_errs

def get_alpha2_data(plotting=False):

    #weaver
    weaver_alpha_high_mass_norm_val = np.array([-0.46, -0.61, -0.63, -0.51, -0.43, 0.07])
    weaver_alpha_high_mass_norm_errl = np.array([0.46, 0.39, 0.44, 0.62, 0.31, 0.60])
    weaver_alpha_high_mass_norm_errh = np.array([0.5, 0.46, 0.48, 0.62, 0.37, 0.58])
    weaver_alpha_high_mass_norm_errs = (weaver_alpha_high_mass_norm_errl+weaver_alpha_high_mass_norm_errh)/2#np.maximum(weaver_alpha_high_mass_norm_errl, weaver_alpha_high_mass_norm_errh)

    weaver_redshift_lower_bin_edge_lowz = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0])
    weaver_redshift_upper_bin_edge_lowz = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5])
    weaver_redshift_midpoint_lowz = (weaver_redshift_lower_bin_edge_lowz + weaver_redshift_upper_bin_edge_lowz)/2

    #cont
    cont_alpha_high_mass_norm_val = np.array([-0.28, -0.28, -0.28])
    cont_alpha_high_mass_norm_errl = np.array([0.07, 0.07, 0.07])
    cont_alpha_high_mass_norm_errh = np.array([0.07, 0.07, 0.07])
    cont_alpha_high_mass_norm_errs = (cont_alpha_high_mass_norm_errl+cont_alpha_high_mass_norm_errh)/2#np.maximum(cont_alpha_high_mass_norm_errl, cont_alpha_high_mass_norm_errh)

    cont_redshift_anchor_points = np.array([0.2, 1.6, 3.0])

    #mcleod
    mcleod_alpha_high_mass_val = np.array([-0.20, -0.25, 0.08, -0.07, -0.06, 0.02, 0.35])
    mcleod_alpha_high_mass_errl = np.array([0.2, 0.25, 0.37, 0.26, 0.39, 0.59, 1.06])
    mcleod_alpha_high_mass_errh = np.array([0.2, 0.25, 0.37, 0.26, 0.39, 0.59, 1.06])
    mcleod_alpha_high_mass_errs = (mcleod_alpha_high_mass_errl+mcleod_alpha_high_mass_errh)/2#np.maximum(mcleod_alpha_high_mass_errl, mcleod_alpha_high_mass_errh)

    mcleod_redshift_lower_bin_edge = np.array([0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    mcleod_redshift_upper_bin_edge = np.array([0.06, 0.75, 1.25, 1.75, 2.25, 2.75, 3.75001])
    mcleod_redshift_midpoint = (mcleod_redshift_lower_bin_edge + mcleod_redshift_upper_bin_edge)/2

    train_alpha2 = torch.from_numpy(np.concatenate((weaver_alpha_high_mass_norm_val, cont_alpha_high_mass_norm_val, mcleod_alpha_high_mass_val)))
    train_alpha2_errs = torch.from_numpy(np.concatenate((weaver_alpha_high_mass_norm_errs, cont_alpha_high_mass_norm_errs, mcleod_alpha_high_mass_errs)))
    train_redshift = torch.from_numpy(np.concatenate((weaver_redshift_midpoint_lowz, cont_redshift_anchor_points, mcleod_redshift_midpoint)))

    sorted_redshift_inds = train_redshift.argsort()[:]
    sorted_train_alpha2 = train_alpha2[sorted_redshift_inds]
    sorted_train_alpha2_errs = train_alpha2_errs[sorted_redshift_inds]
    sorted_train_redshift = train_redshift[sorted_redshift_inds]

    if(plotting):

        plt.errorbar(sorted_train_redshift, sorted_train_alpha2, sorted_train_alpha2_errs, fmt='ko')
        plt.ylabel("High Mass Slope $\\alpha_{2}$")
        plt.xlabel("redshift")
        plt.xlim(-0.2, 7.2)

    return sorted_train_redshift, sorted_train_alpha2, sorted_train_alpha2_errs

def log_error_prop(val, errs):

    arr = unumpy.uarray(val, errs)
    log_arr = unumpy.log10(arr)

    log_val = unumpy.nominal_values(log_arr)
    log_err = unumpy.std_devs(log_arr)
    return log_val, log_err