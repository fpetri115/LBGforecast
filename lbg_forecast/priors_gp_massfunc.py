import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
import lbg_forecast.cosmology as cosmo
import emcee
import lbg_forecast.utils as utils

from uncertainties import unumpy

class GPModel(gpytorch.models.ExactGP):

        def __init__(self, train_x, train_y, lengthscale, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(lengthscale[0], lengthscale[1])))

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

        self.param_names = ["$\mathrm{log}_{10}\phi_{1}^{*}$", "$\mathrm{log}_{10}\phi_{2}^{*}$", "$\\alpha_{1}$", "$\\alpha_{2}$", "$\mathrm{log}_{10}\mathcal{M}_{*}$"]
        state_dict_phi1 = torch.load(self.path+'/gp_models/phi1.pth', weights_only=True)
        state_dict_phi2 = torch.load(self.path+'/gp_models/phi2.pth', weights_only=True)
        state_dict_alpha1 = torch.load(self.path+'/gp_models/alpha1.pth', weights_only=True)
        state_dict_alpha2 = torch.load(self.path+'/gp_models/alpha2.pth', weights_only=True)
        state_dict_logm = torch.load(self.path+'/gp_models/logm.pth', weights_only=True)

        sorted_train_redshift_logphi1, sorted_train_logphi1, sorted_train_logphi1_errs = get_phi1_data(plotting=False)
        sorted_train_redshift_logphi2, sorted_train_logphi2, sorted_train_logphi2_errs = get_phi2_data(plotting=False)
        sorted_train_redshift_alpha1, sorted_train_alpha1, sorted_train_alpha1_errs = get_alpha1_data(plotting=False)
        sorted_train_redshift_alpha2, sorted_train_alpha2, sorted_train_alpha2_errs = get_alpha2_data(plotting=False)
        sorted_train_redshift_logm, sorted_train_logm, sorted_train_logm_errs = get_logm_data(plotting=False)

        self.model_phi1 = create_gp_model([0.0, 7.0], sorted_train_logphi1_errs, sorted_train_redshift_logphi1, sorted_train_logphi1)[0]
        self.model_phi1.load_state_dict(state_dict_phi1)
        self.model_phi1.eval()
        self.phi1_test_z = torch.linspace(0, 7, 100)

        self.model_phi2 = create_gp_model([2.0, 7.0], sorted_train_logphi2_errs, sorted_train_redshift_logphi2, sorted_train_logphi2)[0]
        self.model_phi2.load_state_dict(state_dict_phi2)
        self.model_phi2.eval()
        self.phi2_test_z = torch.linspace(0, 3.0, 100)

        self.model_alpha1 = create_gp_model([5.0, 7.0], sorted_train_alpha1_errs, sorted_train_redshift_alpha1, sorted_train_alpha1)[0]
        self.model_alpha1.load_state_dict(state_dict_alpha1)
        self.model_alpha1.eval()
        self.alpha1_test_z = torch.linspace(0, 7.0, 100)

        self.model_alpha2 = create_gp_model([0.0, 3.0], sorted_train_alpha2_errs, sorted_train_redshift_alpha2, sorted_train_alpha2)[0]
        self.model_alpha2.load_state_dict(state_dict_alpha2)
        self.model_alpha2.eval()
        self.alpha2_test_z = torch.linspace(0, 3.0, 100)

        self.model_logm = create_gp_model([0.0, 7.0], sorted_train_logm_errs, sorted_train_redshift_logm, sorted_train_logm)[0]
        self.model_logm.load_state_dict(state_dict_logm)
        self.model_logm.eval()
        self.logm_test_z = torch.linspace(0, 7, 100)

        self.prior_phi1 = self.model_phi1(self.phi1_test_z)
        self.prior_phi2 = self.model_phi2(self.phi2_test_z)
        self.prior_alpha1 = self.model_alpha1(self.alpha1_test_z)
        self.prior_alpha2 = self.model_alpha2(self.alpha2_test_z)
        self.prior_logm = self.model_logm(self.logm_test_z)

        self.priors = [self.prior_phi1, self.prior_phi2, self.prior_alpha1, self.prior_alpha2, self.prior_logm]
        self.train_x = [sorted_train_redshift_logphi1, sorted_train_redshift_logphi2, sorted_train_redshift_alpha1, sorted_train_redshift_alpha2, sorted_train_redshift_logm]
        self.train_y = [sorted_train_logphi1, sorted_train_logphi2, sorted_train_alpha1, sorted_train_alpha2, sorted_train_logm]
        self.train_yerr = [sorted_train_logphi1_errs, sorted_train_logphi2_errs, sorted_train_alpha1_errs, sorted_train_alpha2_errs, sorted_train_logm_errs]
        self.test_x = [self.phi1_test_z, self.phi2_test_z, self.alpha1_test_z, self.alpha2_test_z, self.logm_test_z]

    def mass_function(self, z, logm, sparams):
        """sparams=self.sample_prior(). logm can be array
        """

        logphi1, logphi2, alpha1, alpha2, logm_star = sparams
        logphi1 = np.interp(z, self.phi1_test_z, logphi1)
        logphi2 = np.interp(z, self.phi2_test_z, logphi2)
        alpha1 = np.interp(z, self.alpha1_test_z, alpha1)
        alpha2 = np.interp(z, self.alpha2_test_z, alpha2)
        logm_star = np.interp(z, self.logm_test_z, logm_star)

        mfunc = np.where(np.atleast_1d(z)>3, self.schechter_function(logm, logphi1, logm_star, alpha1), self.double_schechter_function(logm, logphi1, logphi2, alpha1, alpha2, logm_star))

        return mfunc
    
    def lsst_number_density(self, sparams):
        i=0
        n_z=[]
        for z in self.z_grid:
            n_logm = self.mass_function(z, self.logm_grid, sparams)
            n_logm = np.trapz(n_logm, self.logm_grid)
            n_z.append(n_logm)
            i+=1

        return np.trapz(n_z*self.dvdzgrid, self.z_grid)*utils.LSST_AREA_FRACTION/(utils.LSST_AREA_DEG2*utils.DEG2_TO_ARCMIN2)

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
    
    def logpdf(self, x, sparams, norm, prior_bounds=[0.0,7.0,7.0,13]):
        """log10(P(z, logm))"""

        z, logm = x

        if(z < prior_bounds[0] or z > prior_bounds[1]):
            return -np.inf
        
        if(logm < prior_bounds[2] or logm > prior_bounds[3]):
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
        prior_bounds=[0.0,7.0,7,13]

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

        return redshift_samples[burnin:nsamples+burnin], mass_samples[burnin:nsamples+burnin]

    def schechter_function(self, logm, logphi, logm_star, alpha):
        return np.log(10)*(10**logphi)*10**((logm-logm_star)*(alpha+1))*np.exp(-10**(logm-logm_star))
    def double_schechter_function(self, logm, logphi1, logphi2, alpha1, alpha2, logm_star):
        return self.schechter_function(logm, logphi1, logm_star, alpha1) + self.schechter_function(logm, logphi2, logm_star, alpha2)
    
    def sample_prior(self):
        return[self.sample_phi1(), self.sample_phi2(), self.sample_alpha1(), self.sample_alpha2(), self.sample_logm()]
    def sample_prior_mean(self):
        return[self.sample_phi1_mean(), self.sample_phi2_mean(), self.sample_alpha1_mean(), self.sample_alpha2_mean(), self.sample_logm_mean()]
    
    def sample_phi1(self):
        return self.prior_phi1.sample().numpy()
    def sample_phi1_mean(self):
        return self.prior_phi1.mean.detach().numpy()
    
    def sample_phi2(self):
        return self.prior_phi2.sample().numpy()
    def sample_phi2_mean(self):
        return self.prior_phi2.mean.detach().numpy()
    
    def sample_alpha1(self):
        return self.prior_alpha1.sample().numpy()
    def sample_alpha1_mean(self):
        return self.prior_alpha1.mean.detach().numpy()
    
    def sample_alpha2(self):
        return self.prior_alpha2.sample().numpy()
    def sample_alpha2_mean(self):
        return self.prior_alpha2.mean.detach().numpy()
    
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
            f, ax = plt.subplots(5, 1, figsize=(10, 20), sharex=True)

            indx = 0
            for plot in ax:
                
                prior = self.priors[indx]
                train_x = self.train_x[indx]
                train_y = self.train_y[indx]
                train_y_err = self.train_yerr[indx]
                test_x = self.test_x[indx]
                
                anchor_points = np.array([0.2, 1.6, 3.0])
                #find leja 2020 data
                anchor_indexes = []
                for a in anchor_points:
                    anchor_indexes.append(np.where(train_x.numpy() == a)[0][0])

                mcleod_redshift_lower_bin_edge = np.array([0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
                mcleod_redshift_upper_bin_edge = np.array([0.06, 0.75, 1.25, 1.75, 2.25, 2.75, 3.75001])
                mcleod_points = (mcleod_redshift_lower_bin_edge + mcleod_redshift_upper_bin_edge)/2

                mcleod_indexes = []
                for p in mcleod_points:
                    mcleod_indexes.append(np.where(train_x.numpy() == p)[0][0])



                # Get upper and lower confidence bounds
                lower, upper = prior.confidence_region()
                # Plot training data

                if(indx==0):
                    plot.errorbar(np.delete(train_x.numpy(), np.concatenate((anchor_indexes, mcleod_indexes))), np.delete(train_y.numpy(), np.concatenate((anchor_indexes, mcleod_indexes))), yerr=np.delete(train_y_err, np.concatenate((anchor_indexes, mcleod_indexes))), fmt='d', color='grey', capsize=5, ms=12, label='Weaver et al. (2023)', elinewidth=3, ecolor='k')
                    plot.errorbar(anchor_points, train_y.numpy()[anchor_indexes], yerr=train_y_err[anchor_indexes], fmt='o', color='brown', capsize=5, ms=12, label='Leja et al. (2020)', ecolor='k', elinewidth=3)
                    plot.errorbar(mcleod_points, train_y.numpy()[mcleod_indexes], yerr=train_y_err[mcleod_indexes], fmt='v', color='purple', capsize=5, ms=12, label='Mcleod et al. (2021)', ecolor='k', elinewidth=3)

                    # Plot predictive means as blue line
                    plot.plot(test_x.numpy(), prior.mean, 'b', lw=5, label='Gaussian Process Mean')
                    # Shade between the lower and upper confidence bounds
                    plot.fill_between(test_x.numpy(), lower, upper, alpha=0.25, label='Gaussian Process 2$\sigma$ Confidence')

                    plot.legend(fontsize=14)

                else:
                    plot.errorbar(np.delete(train_x.numpy(), np.concatenate((anchor_indexes, mcleod_indexes))), np.delete(train_y.numpy(), np.concatenate((anchor_indexes, mcleod_indexes))), yerr=np.delete(train_y_err, np.concatenate((anchor_indexes, mcleod_indexes))), fmt='d', color='grey', capsize=5, ms=12, elinewidth=3, ecolor='k')
                    plot.errorbar(anchor_points, train_y.numpy()[anchor_indexes], yerr=train_y_err[anchor_indexes], fmt='o', color='brown', capsize=5, ms=12, ecolor='k', elinewidth=3)
                    plot.errorbar(mcleod_points, train_y.numpy()[mcleod_indexes], yerr=train_y_err[mcleod_indexes], fmt='v', color='purple', capsize=5, ms=12, label='Mcleod et al. (2021)', ecolor='k', elinewidth=3)

                    # Plot predictive means as blue line
                    plot.plot(test_x.numpy(), prior.mean, 'b', lw=5)
                    # Shade between the lower and upper confidence bounds
                    plot.fill_between(test_x.numpy(), lower, upper, alpha=0.25)

                #ax.legend(['Observed Data', 'Mean', 'Confidence'])
                plot.set_ylabel(self.param_names[indx], fontsize=32)


                if(indx==1 or indx==3):
                    plot.set_xlim(-0.2, 7.2)
                    plot.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
                else:
                    plot.set_xlim(-0.2, 7.2)
                    plot.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])

                plot.tick_params('x', labelsize=20, direction='in', width=2, size=7, top=True)
                plot.tick_params('y', labelsize=20, direction='in', width=2, size=7, right=True)
                plot.minorticks_on()
                plot.tick_params(axis='x', which='minor', direction='in', size=5, top=True)
                plot.tick_params(axis='y', which='minor', direction='in', size=5, right=True)
                indx+=1

            plot.set_xlabel("Redshift", fontsize=32)
            plt.tight_layout()

def create_gp_model(lengthscale, errors, train_x, train_y):

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.square(errors), learn_additional_noise=False, noise_constraint=gpytorch.constraints.GreaterThan(0.0))
    model = GPModel(train_x, train_y, lengthscale, likelihood)

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

def gp(train_x, train_y, train_y_errs, test_x, lengthscale, lr, training_iter, ylabel, name):

    #initialise
    model, likelihood = create_gp_model(lengthscale, train_y_errs, train_x, train_y)

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

    #weaver
    weaver_low_mass_norm_val = np.array([0.73, 0.66, 0.84, 0.72, 0.29, 0.27, 0.24, 0.21, 0.20, 0.14, 0.06, 0.03])
    weaver_low_mass_norm_errl = np.array([0.27, 0.27, 0.31, 0.23, 0.11, 0.08, 0.02, 0.03, 0.03, 0.03, 0.02, 0.02])
    weaver_low_mass_norm_errh = np.array([0.25, 0.22, 0.20, 0.15, 0.11, 0.09, 0.03, 0.03, 0.03, 0.04, 0.03, 0.03])
    weaver_low_mass_norm_errs = np.minimum(weaver_low_mass_norm_errl, weaver_low_mass_norm_errh)

    weaver_redshift_lower_bin_edge = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5])
    weaver_redshift_upper_bin_edge = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5])
    weaver_redshift_midpoint = (weaver_redshift_lower_bin_edge + weaver_redshift_upper_bin_edge)/2

    #cont
    log_cont_low_mass_norm_val = np.array([-2.89, -3.29, -3.51])
    cont_low_mass_norm_log_errl = np.array([0.04, 0.03, 0.03])
    cont_low_mass_norm_log_errh = np.array([0.03, 0.03, 0.03])
    cont_low_mass_norm_log_errs = np.minimum(cont_low_mass_norm_log_errl, cont_low_mass_norm_log_errh)

    cont_redshift_anchor_points = np.array([0.2, 1.6, 3.0])

    #mcleod
    log_mcleod_low_mass_norm_val = np.array([-3.03, -3.11, -3.07, -3.32, -3.51, -3.50, -3.74])
    mcleod_low_mass_norm_log_errl = np.array([0.13, 0.11, 0.14, 0.14, 0.22, 0.28, 0.38])
    mcleod_low_mass_norm_log_errh = np.array([0.10, 0.09, 0.11, 0.10, 0.15, 0.17, 0.20])
    mcleod_low_mass_norm_log_errs = np.minimum(mcleod_low_mass_norm_log_errl, mcleod_low_mass_norm_log_errh)

    mcleod_redshift_lower_bin_edge = np.array([0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    mcleod_redshift_upper_bin_edge = np.array([0.06, 0.75, 1.25, 1.75, 2.25, 2.75, 3.75001])
    mcleod_redshift_midpoint = (mcleod_redshift_lower_bin_edge + mcleod_redshift_upper_bin_edge)/2

    weaver_arr = unumpy.uarray(weaver_low_mass_norm_val, weaver_low_mass_norm_errs)
    log_weaver_arr = unumpy.log10((10**-3)*weaver_arr)

    log_weaver_val = unumpy.nominal_values(log_weaver_arr)
    log_weaver_err = unumpy.std_devs(log_weaver_arr)

    train_logphi1 = torch.from_numpy(np.concatenate((log_weaver_val, log_cont_low_mass_norm_val, log_mcleod_low_mass_norm_val)))
    train_logphi1_errs = torch.from_numpy(np.concatenate((log_weaver_err, cont_low_mass_norm_log_errs, mcleod_low_mass_norm_log_errs)))
    train_redshift = torch.from_numpy(np.concatenate((weaver_redshift_midpoint, cont_redshift_anchor_points, mcleod_redshift_midpoint)))

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

def get_phi2_data(plotting=False):

    #weaver
    weaver_high_mass_norm_val = np.array([1.09, 0.83, 0.66, 0.34, 0.64, 0.27])
    weaver_high_mass_norm_errl = np.array([0.54, 0.43, 0.42, 0.25, 0.16, 0.12])
    weaver_high_mass_norm_errh = np.array([0.5, 0.37, 0.34, 0.30, 0.13, 0.12])
    weaver_high_mass_norm_errs = np.maximum(weaver_high_mass_norm_errl, weaver_high_mass_norm_errh)

    weaver_redshift_lower_bin_edge_lowz = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0])
    weaver_redshift_upper_bin_edge_lowz = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5])
    weaver_redshift_midpoint_lowz = (weaver_redshift_lower_bin_edge_lowz + weaver_redshift_upper_bin_edge_lowz)/2

    #cont
    log_cont_high_mass_norm_val = np.array([-2.44, -3.08, -4.14])
    cont_high_mass_norm_log_errl = np.array([0.02, 0.03, 0.11])
    cont_high_mass_norm_log_errh = np.array([0.02, 0.02, 0.10])
    cont_high_mass_norm_log_errs = np.maximum(cont_high_mass_norm_log_errl, cont_high_mass_norm_log_errh)

    cont_redshift_anchor_points = np.array([0.2, 1.6, 3.0])

    #mcleod
    log_mcleod_high_mass_norm_val = np.array([-2.37, -2.63, -2.67, -2.83, -3.05, -3.28, -4.08])
    mcleod_high_mass_norm_log_errl = np.array([0.04, 0.05, 0.07, 0.04, 0.07, 0.10, 0.33])
    mcleod_high_mass_norm_log_errh = np.array([0.03, 0.05, 0.06, 0.04, 0.06, 0.08, 0.18])
    mcleod_high_mass_norm_log_errs = np.maximum(mcleod_high_mass_norm_log_errl, mcleod_high_mass_norm_log_errh)

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

def get_alpha1_data(plotting=False):

    #weaver
    weaver_alpha_low_mass_norm_val = np.array([-1.42, -1.39, -1.32, -1.33, -1.48, -1.46])#, -1.46, -1.46, -1.46, -1.46, -1.46, -1.46])
    weaver_alpha_low_mass_norm_errl = np.array([0.06, 0.07, 0.06, 0.05, 0.09, 0.06])#, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    weaver_alpha_low_mass_norm_errh = np.array([0.05, 0.05, 0.04, 0.05, 0.07, 0.05])#, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    weaver_alpha_low_mass_norm_errs = np.maximum(weaver_alpha_low_mass_norm_errl, weaver_alpha_low_mass_norm_errh)

    weaver_redshift_lower_bin_edge = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0])#, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5])
    weaver_redshift_upper_bin_edge = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5])#, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5])
    weaver_redshift_midpoint = (weaver_redshift_lower_bin_edge + weaver_redshift_upper_bin_edge)/2

    #cont
    cont_alpha_low_mass_norm_val = np.array([-1.48, -1.48, -1.48])
    cont_alpha_low_mass_norm_errl = np.array([0.02, 0.02, 0.02])
    cont_alpha_low_mass_norm_errh = np.array([0.01, 0.01, 0.01])
    cont_alpha_low_mass_norm_errs = np.maximum(cont_alpha_low_mass_norm_errl, cont_alpha_low_mass_norm_errh)

    cont_redshift_anchor_points = np.array([0.2, 1.6, 3.0])

    #mcleod
    mcleod_alpha_low_mass_val = np.array([-1.45, -1.49, -1.49, -1.60, -1.63, -1.66, -1.76])
    mcleod_alpha_low_mass_errl = np.array([0.04, 0.03, 0.05, 0.06, 0.09, 0.10, 0.10])
    mcleod_alpha_low_mass_errh = np.array([0.04, 0.03, 0.05, 0.06, 0.09, 0.10, 0.10])
    mcleod_alpha_low_mass_errs = np.maximum(mcleod_alpha_low_mass_errl, mcleod_alpha_low_mass_errh)

    mcleod_redshift_lower_bin_edge = np.array([0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    mcleod_redshift_upper_bin_edge = np.array([0.06, 0.75, 1.25, 1.75, 2.25, 2.75, 3.75001])
    mcleod_redshift_midpoint = (mcleod_redshift_lower_bin_edge + mcleod_redshift_upper_bin_edge)/2

    train_alpha1 = torch.from_numpy(np.concatenate((weaver_alpha_low_mass_norm_val, cont_alpha_low_mass_norm_val, mcleod_alpha_low_mass_val)))
    train_alpha1_errs = torch.from_numpy(np.concatenate((weaver_alpha_low_mass_norm_errs, cont_alpha_low_mass_norm_errs, mcleod_alpha_low_mass_errs)))
    train_redshift = torch.from_numpy(np.concatenate((weaver_redshift_midpoint, cont_redshift_anchor_points, mcleod_redshift_midpoint)))

    sorted_redshift_inds = train_redshift.argsort()[:]
    sorted_train_alpha1 = train_alpha1[sorted_redshift_inds]
    sorted_train_alpha1_errs = train_alpha1_errs[sorted_redshift_inds]
    sorted_train_redshift = train_redshift[sorted_redshift_inds]

    if(plotting):

        plt.errorbar(sorted_train_redshift, sorted_train_alpha1, sorted_train_alpha1_errs, fmt='ko')
        plt.ylabel("Low Mass Slope $\\alpha_{1}$")
        plt.xlabel("redshift")

    return sorted_train_redshift, sorted_train_alpha1, sorted_train_alpha1_errs

def get_alpha2_data(plotting=False):

    #weaver
    weaver_alpha_high_mass_norm_val = np.array([-0.46, -0.61, -0.63, -0.51, -0.43, 0.07])
    weaver_alpha_high_mass_norm_errl = np.array([0.46, 0.39, 0.44, 0.62, 0.31, 0.60])
    weaver_alpha_high_mass_norm_errh = np.array([0.5, 0.46, 0.48, 0.62, 0.37, 0.58])
    weaver_alpha_high_mass_norm_errs = np.maximum(weaver_alpha_high_mass_norm_errl, weaver_alpha_high_mass_norm_errh)

    weaver_redshift_lower_bin_edge_lowz = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0])
    weaver_redshift_upper_bin_edge_lowz = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5])
    weaver_redshift_midpoint_lowz = (weaver_redshift_lower_bin_edge_lowz + weaver_redshift_upper_bin_edge_lowz)/2

    #cont
    cont_alpha_high_mass_norm_val = np.array([-0.28, -0.28, -0.28])
    cont_alpha_high_mass_norm_errl = np.array([0.07, 0.07, 0.07])
    cont_alpha_high_mass_norm_errh = np.array([0.07, 0.07, 0.07])
    cont_alpha_high_mass_norm_errs = np.maximum(cont_alpha_high_mass_norm_errl, cont_alpha_high_mass_norm_errh)

    cont_redshift_anchor_points = np.array([0.2, 1.6, 3.0])

    #mcleod
    mcleod_alpha_high_mass_val = np.array([-0.20, -0.25, 0.08, -0.07, -0.06, 0.02, 0.35])
    mcleod_alpha_high_mass_errl = np.array([0.2, 0.25, 0.37, 0.26, 0.39, 0.59, 1.06])
    mcleod_alpha_high_mass_errh = np.array([0.2, 0.25, 0.37, 0.26, 0.39, 0.59, 1.06])
    mcleod_alpha_high_mass_errs = np.maximum(mcleod_alpha_high_mass_errl, mcleod_alpha_high_mass_errh)

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

def get_logm_data(plotting=False):

    #weaver
    weaver_logm_val = np.array([10.89, 10.96, 11.02, 11.00, 10.86, 10.78, 10.97, 10.83, 10.46, 10.30, 10.14, 10.18])
    weaver_logm_errl = np.array([0.14, 0.10, 0.09, 0.11, 0.08, 0.14, 0.07, 0.09, 0.06, 0.10, 0.12, 0.27])
    weaver_logm_errh = np.array([0.14, 0.10, 0.08, 0.07, 0.07, 0.16, 0.06, 0.11, 0.09, 0.10, 0.10, 0.37])
    weaver_logm_errs = np.maximum(weaver_logm_errl, weaver_logm_errh)

    weaver_redshift_lower_bin_edge = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5])
    weaver_redshift_upper_bin_edge = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5])
    weaver_redshift_midpoint = (weaver_redshift_lower_bin_edge + weaver_redshift_upper_bin_edge)/2

    #cont
    cont_logm_val = np.array([10.79, 10.88, 10.84])
    cont_logm_errl = np.array([0.02, 0.02, 0.04])
    cont_logm_errh = np.array([0.02, 0.02, 0.04])
    cont_logm_errs = np.maximum(cont_logm_errl, cont_logm_errh)

    cont_redshift_anchor_points = np.array([0.2, 1.6, 3.0])

    #mcleod
    mcleod_logm_val = np.array([10.60, 10.64, 10.51, 10.54, 10.56, 10.55, 10.64])
    mcleod_logm_errl = np.array([0.05, 0.06, 0.07, 0.05, 0.07, 0.11, 0.17])
    mcleod_logm_errh = np.array([0.05, 0.06, 0.07, 0.05, 0.07, 0.11, 0.17])
    mcleod_logm_errs = np.maximum(mcleod_logm_errl, mcleod_logm_errh)

    mcleod_redshift_lower_bin_edge = np.array([0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
    mcleod_redshift_upper_bin_edge = np.array([0.06, 0.75, 1.25, 1.75, 2.25, 2.75, 3.75001])
    mcleod_redshift_midpoint = (mcleod_redshift_lower_bin_edge + mcleod_redshift_upper_bin_edge)/2

    train_logm = torch.from_numpy(np.concatenate((weaver_logm_val, cont_logm_val, mcleod_logm_val)))
    train_logm_errs = torch.from_numpy(np.concatenate((weaver_logm_errs, cont_logm_errs, mcleod_logm_errs)))
    train_redshift = torch.from_numpy(np.concatenate((weaver_redshift_midpoint, cont_redshift_anchor_points, mcleod_redshift_midpoint)))

    sorted_redshift_inds = train_redshift.argsort()[:]
    sorted_train_logm = train_logm[sorted_redshift_inds]
    sorted_train_logm_errs = train_logm_errs[sorted_redshift_inds]
    sorted_train_redshift = train_redshift[sorted_redshift_inds]

    if(plotting):
        plt.errorbar(sorted_train_redshift, sorted_train_logm, sorted_train_logm_errs, fmt='ko')
        plt.ylabel("Crossover $\mathrm{LogM}_{*}$")
        plt.xlabel("redshift")

    return sorted_train_redshift, sorted_train_logm, sorted_train_logm_errs
