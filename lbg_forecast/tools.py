import numpy as np
import matplotlib.pyplot as plt
import lbg_forecast.sps as sps
import lbg_forecast.hyperparams as hyp
import lbg_forecast.popmodel as pop

# simulate photometry for ngalaxies given some hyperparameters
# returns tuple of 2 arrays
# first element is a ngalaxies x number of filters array containing photometry for the sample
# second element is a ngalaxies x number of SPS parameters containing SPS parameters for each source/galaxy
def simulate_photometry(ngalaxies, hyperparams, dust_type=2, imf_type=2, filters='lsst', show_sfh=False):

    #Define SPS Model
    sps_model = sps.initialise_sps_model(sfh_type=3, dust_type=dust_type, imf_type=imf_type)

    i = 0
    photometry = []
    sps_parameters = []
    while(i < ngalaxies):

        #draw sps parameters for a galaxy and send to fsps
        source = pop.galaxy_population_model(hyperparams)
        sps_parameters.append(source)
        sps.update_sps_model_dpl(sps_model, source, plot=show_sfh)

        #generate photometry for source
        photometry.append(sps.simulate_photometry_fsps(sps_model, logmass=source[-1], filters=filters))

        i+=1
        if(i%100 == 0):
            print(i)

    photometry = np.vstack(np.asarray(photometry))
    sps_parameters = np.vstack(np.asarray(sps_parameters))

    return photometry, sps_parameters

def calculate_colours(photometry):
    
    photo1 = photometry[:,:-1]
    photo2 = photometry[:,1:]
    colours = photo1 - photo2

    return colours







#OLD but has spectra plotting code!
def _simulate_sample_photometry_dpl(nsamples, spectra=False):

    #Define SPS Model
    sps_model = sps.initialise_sps_model(sfh_type=3, dust_type=2)
    bounds = hyp.define_hyperparameter_bounds()
    hyperparams = hyp.sample_hyper_parameters(bounds)
    i = 0
    photo_data = []
    source_data = []
    while(i <  nsamples):

        #Update Model and draw Priors
        sps_params = pop.galaxy_population_model_dpl(hyperparams)
        sps.update_sps_model_dpl(sps_model, sps_params)

        #Generate Photometry
        photo_data.append(sps.simulate_photometry_fsps(sps_model, logmass=sps_params[15], filters='lsst'))
        source_data.append(sps_params)

        #Plot Spectra
        if(spectra):
            spectrum = sps.simulate_sed(sps_model, sps_params)
            
            sps.plot_sed(spectrum, scaley = 16, xmin=2000, xmax=12000, ymin=0,
                ymax=1.4, xsize=20, ysize=10, 
                fontsize=32, log=False, c = 'k')
            
            sps.plot_lsst_filters(factor=1)

        i+=1

    return [np.asarray(photo_data), np.asarray(source_data), np.asarray(hyperparams)]

