





import numpy as np
import scipy as sp
from theta.costfunctions import logarithmic





class KullbackLeibler():
    """Empirical  Kullback Leibler divergence.

    """
    def __init__(self, model):
        self.model = model
    
    def cost(self, x, *y):
        pdf, bins = np.histogram(x, bins='auto', density=True)
        for i in range(bins.size-1):
            bins[i] = 0.5*(bins[i]+bins[i+1])
        bins = np.delete(bins,bins.size-1)    
        pdf = pdf.reshape(1,pdf.size)    
        bins = bins.reshape(1,bins.size)
        return np.sum(sp.special.kl_div(pdf, self.model(bins)))

    
    

class LogLikelyhood():
    """Empirical  Loglikelyhood.

    """
    
    def __init__(self, model):
        self.model = model
    
    def cost(self, x, *y):
        pdf, bins = np.histogram(x, bins='auto', density=True)
        for i in range(bins.size-1):
            bins[i] = 0.5*(bins[i]+bins[i+1])
        bins = np.delete(bins,bins.size-1)    
        pdf = pdf.reshape(1,pdf.size)    
        bins = bins.reshape(1,bins.size)
        return -np.sum(np.log(self.model(bins)))    




class sparselog():
    """Sparse Loglikelyhood.

    """
    def __init__(self, model, l=500):
        self.model = model
        self.l = l

    def cost(self, x, *y):
        q = self.model.q
        w = self.model.w
        #c1 = np.sqrt(np.sum(q**2)) # Frobenius norm
        #c2 = np.sqrt(np.sum(w**2)) # Frobenius norm
        c1 = np.sum(np.abs(q)) # L1 norm
        c2 = np.sum(np.abs(w)) # L1 norm
        return logarithmic.cost(x) + self.l*c1 + self.l*c2
    



class sparseKL():
    """Sparse KL divergence.

    """
    def __init__(self, model, l=0.3):
        self.model = model
        self.l = l

    def cost(self, x, *y):
        q = self.model.q
        w = self.model.w
        #c1 = np.sqrt(np.sum(q**2)) # Frobenius norm
        #c2 = np.sqrt(np.sum(w**2)) # Frobenius norm
        c1 = np.sum(np.abs(q)) # L1 norm
        c2 = np.sum(np.abs(w)) # L1 norm
        kl = KullbackLeibler(self.model)
        return kl.cost(x) + self.l*c1 + self.l*c2

    

