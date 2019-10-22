import sys
sys.path.append('../../main')

from solvers import CMA_es, AMSGrad
from costfunctions import KullbackLeibler, LogLikelyhood
from theta.costfunctions import logarithmic
from theta.minimizer import CMA

import numpy as np
from theta import rtbm
import matplotlib.pyplot as plt



# Generating Data

def mixture(n):
    """ Mixture of a Laplace distribution with triangular tails.
    
    """
    v = np.zeros(n)
    u = np.random.random_sample(n)
    for i in range(n):
        if u[i] < 0.2:
            v[i] = np.random.triangular(-5, -2, 1)
        elif u[i] < 0.8:
            v[i] = np.random.laplace(0, 0.5)
        else:
            v[i] = np.random.triangular(-1, 2, 5)
    return v


n = 5000
data = mixture(n)
data = data.reshape(1, data.size)

# RTBM model
model = rtbm.RTBM(1, 2, random_bound=1, init_max_param_bound=60)

# Training
#minim = CMA_es(KullbackLeibler(data), model, data, validation=0., empirical_cost=False)
#minim = AMSGrad(KullbackLeibler(model), model, data, empirical_cost=True)
minim = CMA(True)

#solution = minim.train(popsize=30, m2=20)  # this is for CMA_es
#solution = minim.train()                  # this is for AMSGrad
solution = minim.train(KullbackLeibler(data), model, data, tolfun=1e-3)


# Plotting

figure = plt.axes()
figure.hist(data.flatten(), bins=50, density=True, color='grey')

x = np.linspace(figure.get_xlim()[0], figure.get_xlim()[1], 100).reshape(1,100)

plt.plot(x.flatten(), model(x).flatten(), color='blue')
plt.savefig('cma-laplace-triangular.png')
plt.show()
