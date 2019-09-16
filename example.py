from solvers import CMA_es, AMSGrad
from costfunctions import KullbackLeibler
import numpy as np
from theta import rtbm
from theta.minimizer import CMA
from theta.costfunctions import logarithmic
import matplotlib.pyplot as plt
import time


# Gaussian Mixture data

def gaussian_mixture(n):
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


# Mixture

def mix(n):
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

#data = np.load('adhoc_sampling.npy')
#data = np.random.laplace(0, 1, n)
#data = np.random.gumbel(0, 1, n)
data = mix(n)

data = data.reshape(1, data.size)


model = rtbm.RTBM(1,2, random_bound=1, init_max_param_bound=60)
#model = rtbm.RTBM(1,2, random_bound=1)
#model.set_parameters(np.array([-0.44837034, -0.2893505 ,  0.01315451 , 0.58871856 , 0.32338273,  1.06699414, 2.48215008 ,-0.56304245 , 1.12495846]))
start = model.get_parameters()

print('~~~~~~ Initial Parameters ~~~~~~~')
print(start)
minim = CMA_es(KullbackLeibler(model), model, data, empirical_cost=True)
#minim = AMSGrad(KullbackLeibler(model), model, data, empirical_cost=True)
#minim = CMA(False)

start_time = time.time()
solution = minim.train(popsize=50, m2=20)
#solution = minim.train()
#solution = minim.train(logarithmic, model, data, tolfun=1e-3)
print('Execution Time: ', (time.time() - start_time), ' seconds' )

# CMA solution
#model.set_parameters(np.array([  0.88350455  , 2.67115935 , -1.16901065, -17.60517144 , -9.68492254 ,11.37612766,  51.66851984 , 11.20321513 ,  9.65735923]))
# AMSGrad solution
#model.set_parameters(np.array([-0.42597353, -0.30269457 ,-0.06580133 , 1.34088376 , 0.26185304  ,2.20791242 ,3.51466088, -0.05647029,  0.99785022]))


print('~~~~~~ Initial Parameters ~~~~~~~')
print(start)
print('~~~~~~ Solution ~~~~~~~')
print(model.get_parameters())

figure = plt.axes()
figure.hist(data.flatten(), bins=50, density=True, color='grey')

x = np.linspace(figure.get_xlim()[0],figure.get_xlim()[1], 100).reshape(1,100)

plt.plot(x.flatten(), model(x).flatten(), color='blue')
plt.savefig('cma-mix.png')
plt.show()

plt.clf()

#solution=np.asarray(solution).reshape(200,2)

#np.save('ams_train.npy', solution)














############################################################################################################

##AMSGrad good solution

#--> starting point:
#np.array([-0.50901503, -0.52941161, -2.10282063, -1.34770152, -6.76243165, 19.94294815, 5.56075328, -4.84839361,  9.48992044])

#--> solution:
#np.array([ 0.30902201,  0.3359585 , -1.28159577, -4.8897243,  -6.29011191 ,18.54314476 ,2.58869069, -0.76330285 , 7.88800213])






#####################################################################
#adhoc.npy

# CMA solution
#[  0.88350455   2.67115935  -1.16901065 -17.60517144  -9.68492254 11.37612766  51.66851984  11.20321513   9.65735923]

# AMSGrad solution
#[-0.42597353 -0.30269457 -0.06580133  1.34088376  0.26185304  2.20791242 3.51466088 -0.05647029  0.99785022]

###########################################################################################3
#laplacian --> np.random.laplace(0, 1, n)

# CMA solution
#[ 0.04463695  0.22138538  0.11908582  7.31878484  3.52332139  1.79883323 37.57671322 14.00099481 10.89189749]



"""

cma_costs = np.load('cma_train.npy')
ams_costs = np.load('ams_train.npy')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(ams_costs[:,0],ams_costs[:,1])
plt.plot(cma_costs[:,0],cma_costs[:,1])
plt.yscale('log')
ax.set_ylabel('KL-Divergence')
ax.set_xlabel('Iterations')
ax.legend(('AMSGrad','CMA-ES'))
plt.savefig('train_costs.png')

plt.show()
"""
