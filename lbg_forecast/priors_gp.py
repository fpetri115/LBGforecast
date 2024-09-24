import gpytorch
import torch
import numpy as np
import lbg_forecast.cosmology as cosmo
from astropy.io import ascii
import matplotlib.pyplot as plt
from lbg_forecast.population_model import truncated_normal as tnormal

class CSFRDModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(CSFRDModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.GreaterThan(7.0)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DustIndexModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(DustIndexModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.GreaterThan(1.0)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class DiffuseDustModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(DiffuseDustModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.GreaterThan(4.0)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class CSFRDPrior():

    def __init__(self):

        state_dict = torch.load('/Users/fpetri/repos/LBGForecast/gp_models/gp_csfrd.pth', weights_only=True)
        self.train_z, self.train_log_csfrd, self.errs, self.log_csfrd, self.err_l, self.err_h = process_training_data_csfrd(get_training_data_csfrd())

        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.square(self.errs),
                                                              learn_additional_noise=False,
                                                                noise_constraint=gpytorch.constraints.GreaterThan(0.0))
        
        self.model = CSFRDModel(train_x=self.train_z, train_y=self.train_log_csfrd, likelihood=self.likelihood)
        self.test_z = torch.linspace(0.001, 7, 200)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.prior = self.model(self.test_z)
        self.systematic_shift = self.calculate_systematic()

        self.behroozi_redshift, self.total_obs_csfr, self.true_csfr = self.get_behroozi19_curves()

    def sample_prior(self):
        return self.reverse_scaling(self.test_z, self.prior.sample()).numpy()
    
    def sample_prior_corrected(self):
        return self.reverse_scaling(self.test_z, self.prior.sample()).numpy() - self.systematic_shift
    
    def get_prior_mean(self):
        return self.reverse_scaling(self.test_z, self.prior.mean.detach()).numpy()
    
    def get_prior_mean_corrected(self):
        return self.reverse_scaling(self.test_z, self.prior.mean.detach()).numpy() - self.systematic_shift

    def reverse_scaling(self, redshift, log_shifted_csfrd):
        return 10**shift_csfrd_inverse(redshift, log_shifted_csfrd, np.flip(self.behroozi_redshift), np.flip(np.log10(self.total_obs_csfr)))
    
    def plot_observed_csfrds(self):

        behroozi19 = self.get_behroozi19_curves()
        with torch.no_grad():

            f, ax = plt.subplots(2, 1, figsize=(9, 10))

            ax1 = ax[0]
            ax2 = ax[1]

            ax1.errorbar(self.train_z, 10**self.log_csfrd, yerr=[self.err_l, self.err_h], fmt="ko", capsize=2, ms=4, lw=1)
            ax1.plot(self.test_z.numpy(), self.get_prior_mean(), lw=3, zorder=1200, c="k")
            lower, upper = self.prior.confidence_region()
            ax1.fill_between(self.test_z.numpy(), self.reverse_scaling(self.test_z, lower), self.reverse_scaling(self.test_z, upper), alpha=0.5, zorder=0, color="purple")
            ax1.plot(behroozi19[0], behroozi19[1], zorder=1100, ls="--", c="orange", lw=3)

            ax1.set_yscale("log")
            ax1.set_xscale('function', functions=(forward, inverse))
            ax1.set_xlabel("Redshift")
            ax1.set_ylabel("Cosmic Star Formation Rate Density")
            ax1.set_xlim(0, 7)
            ax1.set_ylim(0.003, 0.4)

            ax2.errorbar(self.train_z, 10**self.log_csfrd, yerr=[self.err_l, self.err_h], fmt="ko", capsize=2, ms=4, lw=1)
            ax2.plot(self.test_z.numpy(), self.get_prior_mean(), lw=3, zorder=1200, c='k')
            nsamples = 500
            for sample in range(nsamples):
                ax2.plot(self.test_z, self.sample_prior(), c='purple', alpha=0.1, zorder=-1)

            ax2.plot(behroozi19[0], behroozi19[1], zorder=1100, ls="--", c="orange", lw=3)

            ax2.set_yscale("log")
            ax2.set_xscale('function', functions=(forward, inverse))
            ax2.set_xlabel("Redshift")
            ax2.set_ylabel("Cosmic Star Formation Rate Density")
            ax2.set_xlim(0, 7)
            ax2.set_ylim(0.003, 0.4)

    def plot_true_csfrds(self):

        behroozi19 = self.get_behroozi19_curves()
        with torch.no_grad():

            f, ax = plt.subplots(2, 1, figsize=(9, 10))

            ax1 = ax[0]
            ax2 = ax[1]

            ax1.errorbar(self.train_z, 10**self.log_csfrd, yerr=[self.err_l, self.err_h], fmt="ko", capsize=2, ms=4, lw=1)
            ax1.plot(self.test_z.numpy(), self.get_prior_mean_corrected(), lw=3, zorder=1200, c="k")
            lower, upper = self.prior.confidence_region()
            ax1.fill_between(self.test_z.numpy(), self.reverse_scaling(self.test_z, lower)-self.systematic_shift, self.reverse_scaling(self.test_z, upper)-self.systematic_shift, alpha=0.5, zorder=0, color="purple")
            ax1.plot(behroozi19[0], behroozi19[2], zorder=1100, ls="--", c="orange", lw=3)

            ax1.set_yscale("log")
            ax1.set_xscale('function', functions=(forward, inverse))
            ax1.set_xlabel("Redshift")
            ax1.set_ylabel("Cosmic Star Formation Rate Density")
            ax1.set_xlim(0, 7)
            ax1.set_ylim(0.003, 0.4)

            ax2.errorbar(self.train_z, 10**self.log_csfrd, yerr=[self.err_l, self.err_h], fmt="ko", capsize=2, ms=4, lw=1)
            ax2.plot(self.test_z.numpy(), self.get_prior_mean_corrected(), lw=3, zorder=1200, c='k')
            nsamples = 500
            for sample in range(nsamples):
                ax2.plot(self.test_z, self.sample_prior_corrected(), c='purple', alpha=0.1, zorder=-1)

            ax2.plot(behroozi19[0], behroozi19[2], zorder=1100, ls="--", c="orange", lw=3)

            ax2.set_yscale("log")
            ax2.set_xscale('function', functions=(forward, inverse))
            ax2.set_xlabel("Redshift")
            ax2.set_ylabel("Cosmic Star Formation Rate Density")
            ax2.set_xlim(0, 7)
            ax2.set_ylim(0.003, 0.4)

    def plot_combined(self):

        behroozi19 = self.get_behroozi19_curves()
        with torch.no_grad():

            f, ax = plt.subplots(1, 1, figsize=(7, 5))

            ax.errorbar(self.train_z, 10**self.log_csfrd, yerr=[self.err_l, self.err_h], fmt="ko", capsize=2, ms=4, lw=1)
            
            ax.plot(self.test_z.numpy(), self.get_prior_mean(), lw=2, zorder=1200, c="b")
            lower, upper = self.prior.confidence_region()
            ax.fill_between(self.test_z.numpy(), self.reverse_scaling(self.test_z, lower), self.reverse_scaling(self.test_z, upper), alpha=0.5, zorder=0, color="b")
            ax.plot(behroozi19[0], behroozi19[1], zorder=1100, ls="--", c="b", lw=2)


            ax.plot(self.test_z.numpy(), self.get_prior_mean_corrected(), lw=2, zorder=1200, c='r')
            lower, upper = self.prior.confidence_region()
            ax.fill_between(self.test_z.numpy(), self.reverse_scaling(self.test_z, lower)-self.systematic_shift, self.reverse_scaling(self.test_z, upper)-self.systematic_shift, alpha=0.5, zorder=0, color="r")
            ax.plot(behroozi19[0], behroozi19[2], zorder=1100, ls="--", c="r", lw=2)

            ax.set_yscale("log")
            ax.set_xscale('function', functions=(forward, inverse))
            ax.set_xlabel("Redshift")
            ax.set_ylabel("Cosmic Star Formation Rate Density")
            ax.set_xlim(0, 7)
            ax.set_ylim(0.003, 0.4)

    def get_behroozi19_curves(self):

        data = ascii.read("csfr_data/csfrs.dat")  

        scale = np.array(data["Scale"])
        redshift = cosmo.scale_to_z(scale)
        total_obs_csfr = np.array(data["Total_Obs_CSFR"])
        true_csfr = np.array(data["True_CSFR"])

        return redshift, total_obs_csfr, true_csfr
    
    def calculate_systematic(self):

        data = ascii.read("csfr_data/csfrs.dat")  

        scale = np.array(data["Scale"])
        redshift = cosmo.scale_to_z(scale)
        total_obs_csfr = np.array(data["Total_Obs_CSFR"])
        true_csfr = np.array(data["True_CSFR"])
        sys_shift = total_obs_csfr - true_csfr

        return np.interp(self.test_z, np.flip(redshift), np.flip(sys_shift))

class DustIndexPrior():

    def __init__(self):

        state_dict = torch.load('/Users/fpetri/repos/LBGForecast/gp_models/gp_index.pth', weights_only=True)
        self.train_av, self.train_d, self.d_errors, self.d_err_l_pop, self.d_err_h_pop = get_index_training_data()

        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.square(self.d_errors),
                                                              learn_additional_noise=False,
                                                                noise_constraint=gpytorch.constraints.GreaterThan(0.0))
        
        self.model = DustIndexModel(train_x=self.train_av, train_y=torch.zeros_like(self.train_av), likelihood=self.likelihood)
        self.test_av = torch.linspace(0, 2.5, 100)
        self.mean = np.interp(self.test_av, self.train_av, self.train_d)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.prior = self.model(self.test_av)

    def sample_prior(self):
        return self.prior.sample().numpy() + self.mean
    
    def get_prior_mean(self):
        return self.prior.mean + self.mean
    
    def sample_dust_index(self, input_av, nsamples):
        sampled_mean = self.sample_prior()
        output_index = np.atleast_1d(np.interp(input_av, self.test_av, sampled_mean))

        sigma=0.3
        return tnormal(output_index, sigma, -2.2, 0.4, (nsamples, output_index.shape[0]))
    
    def plot_model(self):

        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(2, 1, figsize=(10, 7))

            # Get upper and lower confidence bounds
            lower, upper = self.prior.confidence_region()
            # Plot training data as black stars
            ax[0].errorbar(self.train_av.numpy(), self.train_d.numpy(), yerr=[self.d_err_l_pop, self.d_err_h_pop], fmt='ko', capsize=2)
            # Plot predictive means as blue line
            ax[0].plot(self.test_av.numpy(), self.get_prior_mean(), 'b')
            # Shade between the lower and upper confidence bounds
            ax[0].fill_between(self.test_av.numpy(), lower+self.mean, upper+self.mean, alpha=0.5)
            #ax.legend(['Observed Data', 'Mean', 'Confidence'])

            ax[0].set_xlabel("Av")
            ax[0].set_ylabel("delta")

            nsamples = 1000
            for sample in range(nsamples):
                f_sample = self.sample_prior()
                ax[1].plot(self.test_av, f_sample, c='purple', alpha=0.1)
            
            ax[1].plot(self.test_av, self.get_prior_mean(), zorder=1000, ls='-', c='k')
            ax[1].errorbar(self.train_av.numpy(), self.train_d.numpy(), yerr=[self.d_err_l_pop, self.d_err_h_pop], fmt='ko', capsize=2)

            ax[1].set_xlabel("Av")
            ax[1].set_ylabel("delta")

class DiffuseDustPrior():

    def __init__(self):

        state_dict = torch.load('/Users/fpetri/repos/LBGForecast/gp_models/gp_dust2.pth', weights_only=True)
        self.train_sfr, self.train_av, self.av_errors, self.av_err_l_popsfr, self.av_err_h_popsfr = get_diffuse_dust_training_data()

        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.square(self.av_errors),
                                                              learn_additional_noise=False,
                                                                noise_constraint=gpytorch.constraints.GreaterThan(0.0))
        
        self.model = DustIndexModel(train_x=self.train_av, train_y=torch.zeros_like(self.train_av), likelihood=self.likelihood)
        self.test_sfr = torch.linspace(-5, 3, 100)
        self.mean = np.interp(self.test_sfr, self.train_sfr, self.train_av)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.prior = self.model(self.test_sfr)

    def sample_prior(self):
        return self.prior.sample().numpy() + self.mean
    
    def get_prior_mean(self):
        return self.prior.mean + self.mean
    
    def plot_model(self):

        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(2, 1, figsize=(10, 7))

            # Get upper and lower confidence bounds
            lower, upper = self.prior.confidence_region()
            # Plot training data as black stars
            ax[0].errorbar(self.train_sfr.numpy(), self.train_av.numpy(), yerr=[self.av_err_l_popsfr, self.av_err_h_popsfr], fmt='ko', capsize=2)
            # Plot predictive means as blue line
            ax[0].plot(self.test_sfr.numpy(), self.get_prior_mean(), 'b')
            # Shade between the lower and upper confidence bounds
            ax[0].fill_between(self.test_sfr.numpy(), lower+self.mean, upper+self.mean, alpha=0.5)
            #ax.legend(['Observed Data', 'Mean', 'Confidence'])

            ax[0].set_xlabel("log10SFR")
            ax[0].set_ylabel("Av")

            nsamples = 1000
            for sample in range(nsamples):
                f_sample = self.sample_prior()
                ax[1].plot(self.test_sfr, f_sample, c='purple', alpha=0.1)
            
            ax[1].plot(self.test_sfr, self.get_prior_mean(), zorder=1000, ls='-', c='k')
            ax[1].errorbar(self.train_sfr.numpy(), self.train_av.numpy(), yerr=[self.av_err_l_popsfr, self.av_err_h_popsfr], fmt='ko', capsize=2)

            ax[1].set_xlabel("log10SFR")
            ax[1].set_ylabel("Av")

#data pre-processing
def shift_csfrd(new_redshift, csfrd, redshift, mean):
    return csfrd - np.interp(new_redshift, redshift, mean)
def shift_csfrd_inverse(new_redshift, csfrd, redshift, mean):
    return csfrd + np.interp(new_redshift, redshift, mean)

def process_training_data_csfrd(training_data):


    data = ascii.read("csfr_data/csfrs.dat")  
    scale = np.array(data["Scale"])
    redshift = cosmo.scale_to_z(scale)
    total_obs_csfr = np.array(data["Total_Obs_CSFR"])

    train_redshift = torch.from_numpy(training_data[0])
    train_csfrd = torch.from_numpy(training_data[1])
    train_csfrd_errors_l = torch.from_numpy(training_data[3])
    train_csfrd_errors_h = torch.from_numpy(training_data[4])
    log_train_csfrd_errors_l = torch.from_numpy(training_data[5])
    log_train_csfrd_errors_h = torch.from_numpy(training_data[6])


    log_train_csfrd = torch.log10(train_csfrd)
    log_train_shifted_csfrd = shift_csfrd(train_redshift, log_train_csfrd, np.flip(redshift), np.flip(np.log10(total_obs_csfr)))
    log_train_csfrd_errors = log_train_csfrd_errors_l+log_train_csfrd_errors_h

    return train_redshift, log_train_shifted_csfrd, log_train_csfrd_errors, log_train_csfrd, train_csfrd_errors_l, train_csfrd_errors_h

# Function x**(1/2)
def forward(x):
    return abs(x)**(1/1.5)

def inverse(x):
    return x**(1.5)

def get_training_data_csfrd(plot=False):

    data = ascii.read("csfr_data/obs.txt")
    csfr_rows = data['Type']=="csfr"
    csfr_uv_rows = data['Type']=="csfr_(uv)"
    rows = np.logical_or(csfr_rows, csfr_uv_rows)

    z1 = np.array(data["Z1"])[rows]
    z2 = np.array(data["Z2"])[rows]
    redshift = (z2+z1)/2

    z_err_l = redshift-z1
    z_err_h = z2-redshift

    log_val = np.array(data["Val"])[rows]
    log_err_h = np.array(data["Err_h"])[rows]
    log_err_l = np.array(data["Err_l"])[rows]

    log_val_err_h = log_val + log_err_h
    log_val_err_l = log_val - log_err_l

    val = 10**log_val
    val_err_h = 10**log_val_err_h
    val_err_l = 10**log_val_err_l

    err_h = val_err_h - val
    err_l = val - val_err_l

    train_redshift = redshift
    train_csfrd = val
    train_csfrd_errors = err_h+err_l

    #if(plot):
    #plt.errorbar(redshift, val, yerr=[err_l, err_h], fmt='ko', capsize=2, alpha=0.5)
    #plt.yscale("log")
    #plt.xscale('function', functions=(forward, inverse))
    #plt.xlabel("redshift")
    #plt.ylabel("csfrd")
    #plt.xticks(np.arange(0, 10, 1.0))
    #plt.xlim(0, 10)
    #plt.ylim(0.001, 0.2)

    return train_redshift, train_csfrd, train_csfrd_errors, err_l, err_h, log_err_l, log_err_h, log_val_err_l, log_val_err_h

def extract_from_file(file):
    #https://iopscience.iop.org/article/10.3847/1538-4357/aabf3c
    #and popcosmos

    data = ascii.read(file)
    av = np.array(data['x'])
    n = np.array(data['y'])
    n_err_l_val = np.array(data['yl'])
    n_err_h_val = np.array(data['yh'])
    n_err_l = n - n_err_l_val
    n_err_h = n_err_h_val - n
    n_err = n_err_l + n_err_h

    return n, av, n_err, n_err_l, n_err_h

def get_index_training_data():

    d_pop, av_pop, d_err_pop, d_err_l_pop, d_err_h_pop = extract_from_file("dust_data/popcosmos_data.txt")
    train_d = torch.from_numpy(d_pop)
    train_av = torch.from_numpy(av_pop)
    d_errors = torch.from_numpy(d_err_pop)

    return train_av, train_d, d_errors, d_err_l_pop, d_err_h_pop

def get_diffuse_dust_training_data():

    av_popsfr, sfr_popsfr, av_err_popsfr, av_err_l_popsfr, av_err_h_popsfr = extract_from_file("dust_data/sfr_popcosmos_data.txt")

    train_sfr = torch.from_numpy(sfr_popsfr)
    train_av = torch.from_numpy(av_popsfr)
    av_errors = torch.from_numpy(av_err_popsfr)

    return train_sfr, train_av, av_errors, av_err_l_popsfr, av_err_h_popsfr