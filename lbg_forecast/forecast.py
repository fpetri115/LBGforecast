from lbg_forecast.nz_model import NzModel
import numpy as np

def perform_forecast():

    z_space = np.arange(0, 7, 0.01)
    
    model = NzModel(z_space)

    simulated_u = model.g_data()
    #model.plot_nzs(simulated_u)

    print(model.interloper_fraction(simulated_u[5]))

    