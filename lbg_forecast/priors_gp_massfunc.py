import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import gpytorch


class MassFunctionPrior():

    def __init__(self):

        self.param_names = ["$\mathrm{log}\phi_{1}$", "$\mathrm{log}\phi_{2}$", "$\\alpha_{1}$", "$\\alpha_{2}$", "$\mathrm{log}M$"]

        state_dict_phi1 = torch.load('/Users/fpetri/repos/LBGForecast/gp_models/phi1.pth', weights_only=True)
        state_dict_phi2 = torch.load('/Users/fpetri/repos/LBGForecast/gp_models/phi2.pth', weights_only=True)
        state_dict_alpha1 = torch.load('/Users/fpetri/repos/LBGForecast/gp_models/alpha1.pth', weights_only=True)
        state_dict_alpha2 = torch.load('/Users/fpetri/repos/LBGForecast/gp_models/alpha2.pth', weights_only=True)
        state_dict_logm = torch.load('/Users/fpetri/repos/LBGForecast/gp_models/logm.pth', weights_only=True)

        sorted_train_redshift_logphi1, sorted_train_logphi1, sorted_train_logphi1_errl, sorted_train_logphi1_errh, sorted_train_logphi1_errs = get_phi1_data(plotting=False)
        sorted_train_redshift_logphi2, sorted_train_logphi2, sorted_train_logphi2_errl, sorted_train_logphi2_errh, sorted_train_logphi2_errs = get_phi2_data(plotting=False)
        sorted_train_redshift_alpha1, sorted_train_alpha1, sorted_train_alpha1_errl, sorted_train_alpha1_errh, sorted_train_alpha1_errs = get_alpha1_data(plotting=False)
        sorted_train_redshift_alpha2, sorted_train_alpha2, sorted_train_alpha2_errl, sorted_train_alpha2_errh, sorted_train_alpha2_errs = get_alpha2_data(plotting=False)
        sorted_train_redshift_logm, sorted_train_logm, sorted_train_logm_errl, sorted_train_logm_errh, sorted_train_logm_errs = get_logm_data(plotting=False)

        self.model_phi1 = create_gp_model(0.0, sorted_train_logphi1_errs, sorted_train_redshift_logphi1, sorted_train_logphi1)[0]
        self.model_phi1.load_state_dict(state_dict_phi1)
        self.model_phi1.eval()
        self.phi1_test_z = torch.linspace(0, 7, 100)

        self.model_phi2 = create_gp_model(0.0, sorted_train_logphi2_errs, sorted_train_redshift_logphi2, sorted_train_logphi2)[0]
        self.model_phi2.load_state_dict(state_dict_phi2)
        self.model_phi2.eval()
        self.phi2_test_z = torch.linspace(0, 3, 100)

        self.model_alpha1 = create_gp_model(0.0, sorted_train_alpha1_errs, sorted_train_redshift_alpha1, sorted_train_alpha1)[0]
        self.model_alpha1.load_state_dict(state_dict_alpha1)
        self.model_alpha1.eval()
        self.alpha1_test_z = torch.linspace(0, 3.25, 100)

        self.model_alpha2 = create_gp_model(0.0, sorted_train_alpha2_errs, sorted_train_redshift_alpha2, sorted_train_alpha2)[0]
        self.model_alpha2.load_state_dict(state_dict_alpha2)
        self.model_alpha2.eval()
        self.alpha2_test_z = torch.linspace(0, 3, 100)

        self.model_logm = create_gp_model(0.0, sorted_train_logm_errs, sorted_train_redshift_logm, sorted_train_logm)[0]
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
        self.train_y_errl = [sorted_train_logphi1_errl, sorted_train_logphi2_errl, sorted_train_alpha1_errl, sorted_train_alpha2_errl, sorted_train_logm_errl]
        self.train_y_errh = [sorted_train_logphi1_errh, sorted_train_logphi2_errh, sorted_train_alpha1_errh, sorted_train_alpha2_errh, sorted_train_logm_errh]
        self.test_x = [self.phi1_test_z, self.phi2_test_z, self.alpha1_test_z, self.alpha2_test_z, self.logm_test_z]


    def sample_phi1(self):
        return self.prior_phi1.sample().numpy()
    def sample_phi2(self):
        return self.prior_phi2.sample().numpy()
    def sample_alpha1(self):
        return self.prior_alpha1.sample().numpy()
    def sample_alpha2(self):
        return self.prior_alpha2.sample().numpy()
    def sample_logm(self):
        return self.prior_logm.sample().numpy()

    def plot_confidence(self):

        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(5, 1, figsize=(7, 15), sharex=True)

            indx = 0
            for plot in ax:
                
                prior = self.priors[indx]
                train_x = self.train_x[indx]
                train_y = self.train_y[indx]
                train_y_errl = self.train_y_errl[indx]
                train_y_errh = self.train_y_errh[indx]
                test_x = self.test_x[indx]

                # Get upper and lower confidence bounds
                lower, upper = prior.confidence_region()
                # Plot training data as black stars
                plot.errorbar(train_x.numpy(), train_y.numpy(), yerr=[train_y_errl, train_y_errh], fmt='ko', capsize=2)
                # Plot predictive means as blue line
                plot.plot(test_x.numpy(), prior.mean, 'b')
                # Shade between the lower and upper confidence bounds
                plot.fill_between(test_x.numpy(), lower, upper, alpha=0.5)
                #ax.legend(['Observed Data', 'Mean', 'Confidence'])
                plot.set_ylabel(self.param_names[indx])
                plot.set_xlim(-0.2, 7.2)

                indx+=1

            plot.set_xlabel("redshift")
            plt.tight_layout()


def create_gp_model(lengthscale, errors, train_x, train_y):

    class GPModel(gpytorch.models.ExactGP):

        def __init__(self, train_x, train_y, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.GreaterThan(lengthscale)))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.square(errors), learn_additional_noise=False, noise_constraint=gpytorch.constraints.GreaterThan(0.0))
    model = GPModel(train_x, train_y, likelihood)

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

def gp_plot_confidence(f_preds, test_x, train_x, train_y, train_y_errl, train_yerrh, labely):

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 7))

        # Get upper and lower confidence bounds
        lower, upper = f_preds.confidence_region()
        # Plot training data as black stars
        ax.errorbar(train_x.numpy(), train_y.numpy(), yerr=[train_y_errl, train_yerrh], fmt='ko', capsize=2)
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), f_preds.mean, 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower, upper, alpha=0.5)
        #ax.legend(['Observed Data', 'Mean', 'Confidence'])

        ax.set_xlabel("redshift")
        ax.set_ylabel(labely)

def gp_plot_realisations(f_preds, test_x, train_x, train_y, train_y_errl, train_yerrh, labely):

    with torch.no_grad():

        f, ax = plt.subplots(1, 1, figsize=(10, 7))

        nsamples = 1000
        for sample in range(nsamples):
            f_sample = f_preds.sample()
            #if((f_sample > 0.0).all()):
            ax.plot(test_x, f_sample, c='purple', alpha=0.1)
        ax.plot(test_x, f_preds.mean, zorder=1000, ls='-', c='k')
        ax.errorbar(train_x.numpy(), train_y.numpy(), yerr=[train_y_errl, train_yerrh], fmt='ko', capsize=2)
        
        ax.set_xlabel("redshift")
        ax.set_ylabel(labely)

def gp(train_x, train_y, train_y_errl, train_y_errh, train_y_errs, test_x, lengthscale, lr, training_iter, ylabel, name):

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
    gp_plot_confidence(f_preds, test_x, train_x, train_y, train_y_errl, train_y_errh, ylabel)
    gp_plot_realisations(f_preds, test_x, train_x, train_y, train_y_errl, train_y_errh, ylabel)

    #save
    torch.save(trained_model.state_dict(), 'gp_models/'+name+'.pth')

def error_prop(function, x, xerrl, xerrh):

    value = function(x)
    value_errl = value - function(x-xerrl)
    value_errh = function(x+xerrh) - value
    value_err = function(x+xerrh) - function(x-xerrl)

    return value, value_errl, value_errh, value_err

def cont_to_weaver(x):
    return (10**x)*1000

def weaver_to_cont(x):
    return np.log10(x/1000)

def get_phi1_data(plotting=False):

    weaver_low_mass_norm_val = np.array([0.73, 0.66, 0.84, 0.72, 0.29, 0.27, 0.24, 0.21, 0.20, 0.14, 0.06, 0.03])
    weaver_low_mass_norm_errl = np.array([0.27, 0.27, 0.31, 0.23, 0.11, 0.08, 0.02, 0.03, 0.03, 0.03, 0.02, 0.02])
    weaver_low_mass_norm_errh = np.array([0.25, 0.22, 0.20, 0.15, 0.11, 0.09, 0.03, 0.03, 0.03, 0.04, 0.03, 0.03])
    weaver_redshift_lower_bin_edge = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5])
    weaver_redshift_upper_bin_edge = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5])
    weaver_redshift_midpoint = (weaver_redshift_lower_bin_edge + weaver_redshift_upper_bin_edge)/2

    log_cont_low_mass_norm_val = np.array([-2.89, -3.29, -3.51])
    cont_low_mass_norm_log_errl = np.array([0.04, 0.03, 0.03])
    cont_low_mass_norm_log_errh = np.array([0.03, 0.03, 0.03])
    cont_redshift_anchor_points = np.array([0.2, 1.6, 3.0])

    cont_low_mass_norm_val, cont_low_mass_norm_errl, cont_low_mass_norm_errh, cont_low_mass_norm_errs = error_prop(cont_to_weaver, log_cont_low_mass_norm_val, cont_low_mass_norm_log_errl, cont_low_mass_norm_log_errh)

    zg3_index = np.where(weaver_redshift_midpoint>3.0)[0]
    zg3_redshifts = weaver_redshift_midpoint[zg3_index]
    highz_low_mass_norm_val = weaver_low_mass_norm_val[zg3_index]
    highz_low_mass_norm_errl = weaver_low_mass_norm_val[zg3_index]
    high_zlow_mass_norm_errh = (cont_low_mass_norm_errh[-1] + cont_low_mass_norm_val[-1] - weaver_low_mass_norm_val[zg3_index])


    train_phi1 = torch.from_numpy(np.concatenate((weaver_low_mass_norm_val, cont_low_mass_norm_val)))
    train_phi1_errl = torch.from_numpy(np.concatenate((weaver_low_mass_norm_errl, cont_low_mass_norm_errl)))
    train_phi1_errh = torch.from_numpy(np.concatenate((weaver_low_mass_norm_errh, cont_low_mass_norm_errh)))
    train_phi1_errs = train_phi1_errl + train_phi1_errh
    train_redshift = torch.from_numpy(np.concatenate((weaver_redshift_midpoint, cont_redshift_anchor_points)))

    sorted_redshift_inds = train_redshift.argsort()[:]
    sorted_train_phi1 = train_phi1[sorted_redshift_inds]
    sorted_train_phi1_errl = train_phi1_errl[sorted_redshift_inds]
    sorted_train_phi1_errh = train_phi1_errh[sorted_redshift_inds]
    sorted_train_phi1_errs = sorted_train_phi1_errl + sorted_train_phi1_errh
    sorted_train_redshift = train_redshift[sorted_redshift_inds]

    sorted_train_logphi1, sorted_train_logphi1_errl, sorted_train_logphi1_errh, sorted_train_logphi1_errs = error_prop(weaver_to_cont, sorted_train_phi1, sorted_train_phi1_errl, sorted_train_phi1_errh)

    if(plotting):
        plt.errorbar(weaver_redshift_midpoint, weaver_low_mass_norm_val, yerr=[weaver_low_mass_norm_errl, weaver_low_mass_norm_errh], fmt='ko')


        plt.errorbar(cont_redshift_anchor_points, cont_low_mass_norm_val, yerr=[cont_low_mass_norm_errl, cont_low_mass_norm_errh])

        plt.errorbar(zg3_redshifts, highz_low_mass_norm_val, yerr=[highz_low_mass_norm_errl, high_zlow_mass_norm_errh], fmt='ro')
        plt.ylabel("Low Mass Normalisation $\phi_{1} 10^{-3}\mathrm{Mpc}^{-3}\mathrm{dex}^{-1}$")
        plt.xlabel("redshift")

        flogphi, ax1 = plt.subplots(1, 1, figsize=(10, 7))
        ax1.errorbar(sorted_train_redshift, sorted_train_logphi1, yerr=[sorted_train_logphi1_errl, sorted_train_logphi1_errh], fmt='ko')
        ax1.set_ylabel("Low Mass Normalisation $log_{10}(\phi_{1})$")
        ax1.set_xlabel("redshift")
    
    return sorted_train_redshift, sorted_train_logphi1, sorted_train_logphi1_errl, sorted_train_logphi1_errh, sorted_train_logphi1_errs

def get_phi2_data(plotting=False):

    weaver_high_mass_norm_val = np.array([1.09, 0.83, 0.66, 0.34, 0.64, 0.27])
    weaver_high_mass_norm_errl = np.array([0.54, 0.43, 0.42, 0.25, 0.16, 0.12])
    weaver_high_mass_norm_errh = np.array([0.5, 0.37, 0.34, 0.30, 0.13, 0.12])
    weaver_redshift_lower_bin_edge_lowz = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0])
    weaver_redshift_upper_bin_edge_lowz = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5])
    weaver_redshift_midpoint_lowz = (weaver_redshift_lower_bin_edge_lowz + weaver_redshift_upper_bin_edge_lowz)/2

    log_cont_high_mass_norm_val = np.array([-2.44, -3.08, -4.14])
    cont_high_mass_norm_val = (10**log_cont_high_mass_norm_val)*1000
    cont_high_mass_norm_log_errl = np.array([0.02, 0.03, 0.11])
    cont_high_mass_norm_log_errh = np.array([0.02, 0.02, 0.10])
    cont_high_mass_norm_errl = cont_high_mass_norm_val - (10**(log_cont_high_mass_norm_val - cont_high_mass_norm_log_errl))*1000
    cont_high_mass_norm_errh = (10**(log_cont_high_mass_norm_val + cont_high_mass_norm_log_errh))*1000 - cont_high_mass_norm_val
    cont_redshift_anchor_points = np.array([0.2, 1.6, 3.0])

    train_phi2 = torch.from_numpy(np.concatenate((weaver_high_mass_norm_val, cont_high_mass_norm_val)))
    train_phi2_errl = torch.from_numpy(np.concatenate((weaver_high_mass_norm_errl, cont_high_mass_norm_errl)))
    train_phi2_errh = torch.from_numpy(np.concatenate((weaver_high_mass_norm_errh, cont_high_mass_norm_errh)))
    train_phi2_errs = train_phi2_errl + train_phi2_errh
    train_redshift = torch.from_numpy(np.concatenate((weaver_redshift_midpoint_lowz, cont_redshift_anchor_points)))

    sorted_redshift_inds = train_redshift.argsort()[:]
    sorted_train_phi2 = train_phi2[sorted_redshift_inds]
    sorted_train_phi2_errl = train_phi2_errl[sorted_redshift_inds]
    sorted_train_phi2_errh = train_phi2_errh[sorted_redshift_inds]
    sorted_train_phi2_errs = sorted_train_phi2_errl + sorted_train_phi2_errh
    sorted_train_redshift = train_redshift[sorted_redshift_inds]

    sorted_train_logphi2, sorted_train_logphi2_errl, sorted_train_logphi2_errh, sorted_train_logphi2_errs = error_prop(weaver_to_cont, sorted_train_phi2, sorted_train_phi2_errl, sorted_train_phi2_errh)

    if(plotting):
        plt.errorbar(weaver_redshift_midpoint_lowz, weaver_high_mass_norm_val, yerr=[weaver_high_mass_norm_errl, weaver_high_mass_norm_errh], fmt='ko')

        plt.errorbar(cont_redshift_anchor_points, cont_high_mass_norm_val, yerr=[cont_high_mass_norm_errl, cont_high_mass_norm_errh])
        plt.ylabel("High Mass Normalisation $\phi_{2} 10^{-3}\mathrm{Mpc}^{-3}\mathrm{dex}^{-1}$")
        plt.xlabel("redshift")
        plt.xlim(-0.2, 7.2)

        flogphi, ax2 = plt.subplots(1, 1, figsize=(10, 7))
        ax2.errorbar(sorted_train_redshift, sorted_train_logphi2, yerr=[sorted_train_logphi2_errl, sorted_train_logphi2_errh], fmt='ko')
        ax2.set_ylabel("High Mass Normalisation $log_{10}(\phi_{2})$")
        ax2.set_xlabel("redshift")

    return sorted_train_redshift, sorted_train_logphi2, sorted_train_logphi2_errl, sorted_train_logphi2_errh, sorted_train_logphi2_errs

def get_alpha1_data(plotting=False):

    weaver_alpha_low_mass_norm_val = np.array([-1.42, -1.39, -1.32, -1.33, -1.48, -1.46, -1.46])#, -1.46, -1.46, -1.46, -1.46, -1.46, -1.46])
    weaver_alpha_low_mass_norm_errl = np.array([0.06, 0.07, 0.06, 0.05, 0.09, 0.06, 1e-10])#, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    weaver_alpha_low_mass_norm_errh = np.array([0.05, 0.05, 0.04, 0.05, 0.07, 0.05, 1e-10])#, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    weaver_redshift_lower_bin_edge = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0, 3.0])#, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5])
    weaver_redshift_upper_bin_edge = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.5])#, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5])
    weaver_redshift_midpoint = (weaver_redshift_lower_bin_edge + weaver_redshift_upper_bin_edge)/2

    cont_alpha_low_mass_norm_val = np.array([-1.48, -1.48, -1.48])
    cont_alpha_low_mass_norm_errl = np.array([0.02, 0.02, 0.02])
    cont_alpha_low_mass_norm_errh = np.array([0.01, 0.01, 0.01])
    cont_redshift_anchor_points = np.array([0.2, 1.6, 3.0])

    train_alpha1 = torch.from_numpy(np.concatenate((weaver_alpha_low_mass_norm_val, cont_alpha_low_mass_norm_val)))
    train_alpha1_errl = torch.from_numpy(np.concatenate((weaver_alpha_low_mass_norm_errl, cont_alpha_low_mass_norm_errl)))
    train_alpha1_errh = torch.from_numpy(np.concatenate((weaver_alpha_low_mass_norm_errh, cont_alpha_low_mass_norm_errh)))
    train_alpha1_errs = train_alpha1_errl + train_alpha1_errh
    train_redshift = torch.from_numpy(np.concatenate((weaver_redshift_midpoint, cont_redshift_anchor_points)))

    sorted_redshift_inds = train_redshift.argsort()[:]
    sorted_train_alpha1 = train_alpha1[sorted_redshift_inds]
    sorted_train_alpha1_errl = train_alpha1_errl[sorted_redshift_inds]
    sorted_train_alpha1_errh = train_alpha1_errh[sorted_redshift_inds]
    sorted_train_alpha1_errs = sorted_train_alpha1_errl + sorted_train_alpha1_errh
    sorted_train_redshift = train_redshift[sorted_redshift_inds]

    if(plotting):

        plt.errorbar(weaver_redshift_midpoint, weaver_alpha_low_mass_norm_val, yerr=[weaver_alpha_low_mass_norm_errl, weaver_alpha_low_mass_norm_errh], fmt='ko')

        plt.errorbar(cont_redshift_anchor_points, cont_alpha_low_mass_norm_val, yerr=[cont_alpha_low_mass_norm_errl, cont_alpha_low_mass_norm_errh])
        plt.ylabel("Low Mass Slope $\\alpha_{1}$")
        plt.xlabel("redshift")

    return sorted_train_redshift, sorted_train_alpha1, sorted_train_alpha1_errl, sorted_train_alpha1_errh, sorted_train_alpha1_errs

def get_alpha2_data(plotting=False):

    weaver_alpha_high_mass_norm_val = np.array([-0.46, -0.61, -0.63, -0.51, -0.43, 0.07])
    weaver_alpha_high_mass_norm_errl = np.array([0.46, 0.39, 0.44, 0.62, 0.31, 0.60])
    weaver_alpha_high_mass_norm_errh = np.array([0.5, 0.46, 0.48, 0.62, 0.37, 0.58])
    weaver_redshift_lower_bin_edge_lowz = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0])
    weaver_redshift_upper_bin_edge_lowz = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5])
    weaver_redshift_midpoint_lowz = (weaver_redshift_lower_bin_edge_lowz + weaver_redshift_upper_bin_edge_lowz)/2

    cont_alpha_high_mass_norm_val = np.array([-0.28, -0.28, -0.28])
    cont_alpha_high_mass_norm_errl = np.array([0.07, 0.07, 0.07])
    cont_alpha_high_mass_norm_errh = np.array([0.07, 0.07, 0.07])
    cont_redshift_anchor_points = np.array([0.2, 1.6, 3.0])

    train_alpha2 = torch.from_numpy(np.concatenate((weaver_alpha_high_mass_norm_val, cont_alpha_high_mass_norm_val)))
    train_alpha2_errl = torch.from_numpy(np.concatenate((weaver_alpha_high_mass_norm_errl, cont_alpha_high_mass_norm_errl)))
    train_alpha2_errh = torch.from_numpy(np.concatenate((weaver_alpha_high_mass_norm_errh, cont_alpha_high_mass_norm_errh)))
    train_alpha2_errs = train_alpha2_errl + train_alpha2_errh
    train_redshift = torch.from_numpy(np.concatenate((weaver_redshift_midpoint_lowz, cont_redshift_anchor_points)))

    sorted_redshift_inds = train_redshift.argsort()[:]
    sorted_train_alpha2 = train_alpha2[sorted_redshift_inds]
    sorted_train_alpha2_errl = train_alpha2_errl[sorted_redshift_inds]
    sorted_train_alpha2_errh = train_alpha2_errh[sorted_redshift_inds]
    sorted_train_alpha2_errs = sorted_train_alpha2_errl + sorted_train_alpha2_errh
    sorted_train_redshift = train_redshift[sorted_redshift_inds]

    if(plotting):

        plt.errorbar(weaver_redshift_midpoint_lowz, weaver_alpha_high_mass_norm_val, yerr=[weaver_alpha_high_mass_norm_errl, weaver_alpha_high_mass_norm_errh], fmt='ko')

        plt.errorbar(cont_redshift_anchor_points, cont_alpha_high_mass_norm_val, yerr=[cont_alpha_high_mass_norm_errl, cont_alpha_high_mass_norm_errh])
        plt.ylabel("High Mass Slope $\\alpha_{2}$")
        plt.xlabel("redshift")
        plt.xlim(-0.2, 7.2)

    return sorted_train_redshift, sorted_train_alpha2, sorted_train_alpha2_errl, sorted_train_alpha2_errh, sorted_train_alpha2_errs

def get_logm_data(plotting=False):

    weaver_logm_val = np.array([10.89, 10.96, 11.02, 11.00, 10.86, 10.78, 10.97, 10.83, 10.46, 10.30, 10.14, 10.18])
    weaver_logm_errl = np.array([0.14, 0.10, 0.09, 0.11, 0.08, 0.14, 0.07, 0.09, 0.06, 0.10, 0.12, 0.27])
    weaver_logm_errh = np.array([0.14, 0.10, 0.08, 0.07, 0.07, 0.16, 0.06, 0.11, 0.09, 0.10, 0.10, 0.37])
    weaver_redshift_lower_bin_edge = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5])
    weaver_redshift_upper_bin_edge = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 6.5, 7.5])
    weaver_redshift_midpoint = (weaver_redshift_lower_bin_edge + weaver_redshift_upper_bin_edge)/2

    cont_logm_val = np.array([10.79, 10.88, 10.84])
    cont_logm_errl = np.array([0.02, 0.02, 0.04])
    cont_logm_errh = np.array([0.02, 0.02, 0.04])
    cont_redshift_anchor_points = np.array([0.2, 1.6, 3.0])

    train_logm = torch.from_numpy(np.concatenate((weaver_logm_val, cont_logm_val)))
    train_logm_errl = torch.from_numpy(np.concatenate((weaver_logm_errl, cont_logm_errl)))
    train_logm_errh = torch.from_numpy(np.concatenate((weaver_logm_errh, cont_logm_errh)))
    train_logm_errs = train_logm_errl + train_logm_errh
    train_redshift = torch.from_numpy(np.concatenate((weaver_redshift_midpoint, cont_redshift_anchor_points)))

    sorted_redshift_inds = train_redshift.argsort()[:]
    sorted_train_logm = train_logm[sorted_redshift_inds]
    sorted_train_logm_errl = train_logm_errl[sorted_redshift_inds]
    sorted_train_logm_errh = train_logm_errh[sorted_redshift_inds]
    sorted_train_logm_errs = sorted_train_logm_errl + sorted_train_logm_errh
    sorted_train_redshift = train_redshift[sorted_redshift_inds]

    if(plotting):
        plt.errorbar(weaver_redshift_midpoint, weaver_logm_val, yerr=[weaver_logm_errl, weaver_logm_errh], fmt='ko')

        plt.errorbar(cont_redshift_anchor_points, cont_logm_val, yerr=[cont_logm_errl, cont_logm_errh])
        plt.ylabel("Crossover $\mathrm{LogM}_{*}$")
        plt.xlabel("redshift")

    return sorted_train_redshift, sorted_train_logm, sorted_train_logm_errl, sorted_train_logm_errh, sorted_train_logm_errs
