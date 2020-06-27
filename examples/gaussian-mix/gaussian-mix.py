import sys
sys.path.append('../../main')

from solvers import CMA_es, SGD, RMSProp
from costfunctions import KullbackLeibler, LogLikelyhood, KLdiv

import numpy as np
from theta import rtbm
from theta.costfunctions import logarithmic
import matplotlib.pyplot as plt

#from theta.minimizer import SGD


# Generating Data

def gaussian_mixture(n):
    """ Gaussian Mixture PDF.
    
    """
    v = np.zeros(n)
    u = np.random.random_sample(n)
    for i in range(n):
        if u[i] < 0.6:
            v[i] = np.random.normal(-0.7, 0.3)
        elif u[i] < 0.7:
            v[i] = np.random.normal(0.2, .1)
        else:
            v[i] = np.random.normal(.5, .5)
    return v


n = 10000
data = gaussian_mixture(n)
data = data.reshape(1, data.size)

# RTBM model
model = rtbm.RTBM(1, 2, random_bound=1, diagonal_T=True)

print(model.get_parameters())

# Training
#minim = CMA_es(logarithmic, model, data)
#minim = AMSGrad(KullbackLeibler(model,data), model, data, empirical_cost=True)
#minim = SGD(KullbackLeibler(model,data), model, data, empirical_cost=True)
#minim = SGD(logarithmic, model, data)
minim = RMSProp(logarithmic, model, data)
#minim = SGD()

#solution = minim.train(popsize=20, m2=7)  # this is for CMA_es
solution = minim.train()                  # this is for AMSGrad
#solution = minim.train(logarithmic, model, data)

# Plotting

figure = plt.axes()
figure.hist(data.flatten(), bins=50, density=True, color='grey')

x = np.linspace(figure.get_xlim()[0], figure.get_xlim()[1], 100).reshape(1,100)

plt.plot(x.flatten(), model(x).flatten(), color='blue')
plt.savefig('cma-gaussian-mix.png')
plt.show()
