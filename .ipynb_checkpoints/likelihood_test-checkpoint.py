import numpy as np

class Likelihood():
    
    def __init__(self):
        
        cov = np.eye(8)
        self.invC = np.linalg.inv(cov)
        
    def logp(self, params):
        """Gaussian likelihood for n(z) fixed at mean value"""
        
        Cinv = self.invC
        
        x = params
        x_T = np.transpose(x)
        
        Xsq = -0.5*x_T@Cinv@x
        
        return Xsq
