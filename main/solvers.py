import numpy as np
import scipy as sp
from abc import ABC, abstractmethod
import math
from scipy import optimize
from manifold import Manifold
from theta import rtbm

 


class Solver(ABC):
    """Abstract solver class.

    """
    
    def __init__(self, cost, model, x_data, y_data=None, verbosity=2, batchsize=250, validation=0.33, empirical_cost=False):

        assert (type(model) == rtbm.RTBM), "works only with RTBM objects"

        self.empirical_cost = empirical_cost
        self.input_cost = cost
        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        self.batchsize = batchsize
        self.validation_set, self.training_set = self.set_init(validation)
        self.batch = self.batch_gen()
        self.training_err = 0

        self.man = Manifold(self.model)
        
        
    @abstractmethod
    def train(self, starting_x=None):
        pass

    def set_init(self, validation):
        index = [int(np.random.rand()*self.x_data.size) for j in range(int(self.x_data.size*validation))]
        val = []
        train = []
        for i in range(self.x_data.size):
            if i in index:
                val.append(self.x_data[0,i])
            else:
                train.append(self.x_data[0,i])
        return np.asarray(val), np.asarray(train)        
    
    def batch_gen(self):
        index = [int(np.random.rand()*self.training_set.size) for j in range(self.batchsize)]
        return np.asarray([self.training_set[k] for k in index]).reshape(1,self.batchsize)

    def error(self, set):
        if self.empirical_cost == True:
            return self.input_cost.cost(set.reshape(1,set.size))
        else:
            return self.input_cost.cost(self.model(set.reshape(1,set.size)))
    
    def GL(self):  # generalization loss in %
        return 100*(self.validation_err/self.opt_val_err - 1)

    def Pk(self, strip, k):  # learning progress on the strip of dim k
        return 1000*(np.sum(strip)/(k*np.amin(strip)) - 1)

    def flat_cost(self, params):        
        self.model.set_parameters(params)
        if self.empirical_cost == True:
            return self.input_cost.cost(self.batch)
        else:
            return self.input_cost.cost(self.model(self.batch))

    def cost(self, X):
        params = self.manifold2params(X)
        return self.flat_cost(params)

    def egrad(self, X):
        g = optimize.approx_fprime(self.manifold2params(X), 
                                   self.flat_cost, 1e-8)
        
        pd_grad = np.zeros(shape=(self.model._Nv+self.model._Nh, 
                                  self.model._Nv+self.model._Nh))

        index = self.model._Nv+self.model._Nh
        eucl_grad = g[:index]
        pd_grad[self.model._Nh:,:self.model._Nh] = g[index:index+self.model._Nv*self.model._Nh] #w
        index += self.model._Nv*self.model._Nh
        pd_grad[self.model._Nh:,self.model._Nh:] = g[index:index+self.model._Nv**2] #t
        index += self.model._Nv**2
        inds = np.triu_indices_from(pd_grad[:self.model._Nh,:self.model._Nh])        
        pd_grad[inds] = g[index:] #q
        pd_grad[(inds[1],inds[0])] = g[index:] #q        
        pd_grad[:self.model._Nh,self.model._Nh:] = pd_grad[self.model._Nh:,:self.model._Nh].T #wp    
        return (eucl_grad, pd_grad)

    def params2manifold(self, params):
        nh = self.model._Nh
        nv = self.model._Nv
        n = nv+nh
        # bv & bh
        bias = params[:n]
        index = n
        # w
        tmp = params[index:index+nh*nv]
        w = tmp.reshape(nv,nh)
        index = index+nh*nv
        # t
        tmp = params[index:index+int(nv*(nv+1)/2)]
        t = np.zeros((nv,nv))
        t[np.triu_indices(nv)] = tmp
        t[np.tril_indices(nv)]= t.T[np.tril_indices(nv)]
        index = index+int(nv*(nv+1)/2)
        # q
        tmp = params[index:]
        q = np.zeros((nh,nh))
        q[np.triu_indices(nh)] = tmp
        q[np.tril_indices(nh)]= q.T[np.tril_indices(nh)]

        part1 = np.hstack((q,w.T))
        part2 = np.hstack((w,t))
        A = np.vstack((part1,part2))

        return (bias, A)

    def manifold2params(self, X):
        bias, weights = X
        bv = bias[:self.model._Nv]
        bh = bias[self.model._Nv:]       
        q = (weights[:self.model._Nh,:self.model._Nh])[np.triu_indices(self.model._Nh)]
        if self.model._diagonal_T:
            t = (weights[self.model._Nh:,self.model._Nh:]).diagonal()
        else:
            t = (weights[self.model._Nh:,self.model._Nh:])[np.triu_indices(self.model._Nv)]
        w = weights[self.model._Nh:,:self.model._Nh].flatten()
        return np.concatenate([bv,bh,w,t,q])





    


class CMA_es(Solver):
    """CMA evolution strategy.

    """
    
    def train(self, starting_x=None, popsize=4, m2=3):

        self.batch = self.training_set
        self.popsize = popsize
        self.m2 = m2
        assert (self.m2 < popsize), "too large m2"

        self.init()
        
        if starting_x != None:
            x = starting_x
            self.model.set_parameters(x)
        else:    
            x = self.params2manifold(self.model.get_parameters())

        print('CMA-ES \n', 'Population = ', self.popsize, '\n Mutants = ', self.m2, '\n')
        print('Epochs', '\t\t', 'Cost', '\t\t')

        cost_list = []
            
        vc = 0
        vs = 0
        best_cost = float('Inf')
        best_x = x
        #while True:
        for i in range(1000):
            
            #print('######### Actual Point \n', x)
            cost = self.error(self.validation_set)#self.cost(x)
            if i%5==0:
                cost_list.append([i,cost])
            if cost < best_cost:
                best_cost = cost
                best_x = x
            if(i%5==0):
                print( i, '\t\t', cost,'\t\t')
            # sampling new generation starting from x
            #print('# New Generation')
            v, _= self.new_gen(x)
            #print('\\\\\ V ///// \n', v)
        
            # performing recombination based on best fitted individuals
            #print('## Recombination')
            v_best = self.recombination(v)

            # updating evolution paths
            #print('### Updating internal parameters')
            vc = (1-self.cc)*vc + np.sqrt(self.cc*(2-self.cc)*self.meff)*v_best/self.s  # covariance path
            vs = np.real((1-self.cs)*vs + np.sqrt(self.cs*(2-self.cs)*self.meff)/self.s*np.dot(sp.linalg.sqrtm(sp.linalg.inv(self.C)),v_best))  # sigma path
            #print('\\\\\ Vc ///// \n', vc)
            #print('\\\\\ Vs ///// \n', vs)
            
            # update covariance matrix
            self.C_update(v, vc)
            #print('\\\\\ C ///// \n', self.C)
            
            # update stepsize
            self.s_update(x, vs)
            #print('\\\\\ Sigma ///// \n', self.s)

            # backuping manifold point for future parallel transport
            x_bak = x
            
            # moving to new manifold point
            x = self.man.retr(x, self.params2manifold(v_best))

            self.model.set_parameters(self.manifold2params(x))
            #self.model = model
            #print('\\\\\ New X ///// \n', self.man.manifold2params(x))
            
            # parallel transport
            vc, vs = self.transport(x_bak, x, vc, vs)
            #print('>>>>>>>>> TRansported vs \n', vs)

        self.model.set_parameters(self.manifold2params(best_x))
        #self.model = model
        #return self.model.get_parameters()
        return cost_list
        
    def init(self):
        w = [np.log((self.m2+1)/i) for i in range(1,self.m2+1)]
        self.w = w / np.sum(w)
        self.meff = 1 / np.sum(self.w**2)
        self.N = int(np.sum(self.man.dim))
        self.cc = 4/(self.N+4)
        self.cs = (self.meff+2)/(self.N+self.meff+3)
        self.ccov = 2/(self.meff*(self.N+np.sqrt(2))**2) + (1-1/self.meff)*min(1,(2*self.meff-1)/((self.N+2)**2+self.meff))
        self.ds = 1 + 2*max(0,np.sqrt((self.meff-1)/(self.N+1))-1) + self.cs
        self.s = 0.01   # step size
        self.C = np.identity(self.N)
            
    def new_gen(self, x):
        v = self.s*np.random.multivariate_normal(np.zeros(self.N), self.C, size=self.popsize)
        cost = np.zeros(self.popsize)
        pop = []
        for i in range(self.popsize):
            tmp = self.params2manifold(v[i])
            pop.append(self.man.retr(x, tmp))
            #print('####### Nuovo individuo \n', pop[i])
            #print('---> calcolo costo')
            cost[i] = self.cost(pop[i])
            #print(cost[i])
            while math.isnan(cost[i]) or cost[i] == float('Inf'):
                #print('*****************punto problematico')
                v[i] = self.s*np.random.multivariate_normal(np.zeros(self.N), self.C, size=1)
                tmp = self.params2manifold(v[i])
                #print('####### Nuovo individuo \n', pop[i])
                pop[i] = (self.man.retr(x, tmp))
                #print('---> calcolo costo')
                cost[i] = self.cost(pop[i])
                #print(cost[i])
        v, cost = self.sort(v, cost)
        return (v, cost)
    
    def recombination(self, v):
        best = 0
        for i in range(self.m2):
            best = best + self.w[i]*v[i]
        return best
    
    def sort(self, pop, val):
        tmp = [[pop[i],val[i]] for i in range(self.popsize)]
        tmp.sort(key = lambda x: x[1])
        p = [tmp[i][0] for i in range(self.popsize)]
        v = [tmp[i][1] for i in range(self.popsize)]
        return p,v

    def C_update(self, v, vc):
        tmp = 0
        for i in range(self.m2):
            tmp = tmp + self.w[i]*np.dot(v[i].reshape(self.N,1),v[i].reshape(self.N,1).T)
        self.C = self.ccov*(1-1/self.meff)*(1/self.s**2)*tmp + (1-self.ccov)*self.C + self.ccov/self.meff*np.dot(vc.reshape(self.N,1),vc.reshape(self.N,1).T)

    def s_update(self, x, vs):
        E = np.sqrt(2)*sp.special.gamma((self.N+1)/2)/sp.special.gamma(self.N/2)
        Vs = self.params2manifold(vs)
        #print('>>>>>>>> vs norm \n',self.man.norm(x, Vs))
        self.s = min(self.s*np.exp((self.man.norm(x, Vs)/E-1)*self.cs/self.ds), 0.001)
    
    def transport(self, x1, x2, vc, vs): # check that is all right
        # first transport vc & vs
        #print('#### Transport')
        Vc = self.man.transp(x1, x2, self.params2manifold(vc))
        Vs = self.man.transp(x1, x2, self.params2manifold(vs))
        # now transport C
        #e = self.coord_sys()
        #self.Ctransp(x1, x2, e)
        #print('new C ', self.C)
        return self.manifold2params(Vc), self.manifold2params(Vs)

    def coord_sys(self):
        nv = self.model._Nv
        nh = self.model._Nh
        e = []
        for i in range(nh+nv):
            e.append((np.zeros(nv+nh),np.zeros((nv+nh,nv+nh))))
            e[i][0][i] = 1
        for i in range((nh+nv)**2):
            tmp = np.zeros((nh+nv)**2)
            tmp[i] = 1
            e.append((np.zeros(nv+nh),tmp.reshape(nv+nh,nv+nh)))
        return e

    def Ctransp(self, x1, x2, e):
        et = []
        nv = self.model._Nv
        nh = self.model._Nh
        for i in range(nh+nv+(nh+nv)**2):
            et.append(self.man.transp(x1, x2, e[i]))
            et[i] = self.manifold2params(et[i])
            #print('\\\\\\\\\\\ et[i] \n', et[i])
        newC = np.zeros((self.N, self.N))
        for k in range(self.N):
            for l in range(self.N):
                tmp = 0
                for i in range(self.N):
                    for j in range(self.N):
                        tmp = tmp + et[k][i]*et[l][j]*self.C[i][j]
                newC[k][l] = tmp
        self.C = newC








        



class AMSGrad(Solver):
    """AMSGrad.

    """

    def train(self, starting_x=None):
        
        b1 = 0.9
        b2 = 0.999
        a = 0.01
        eps = 1e-8
        cost_list = []

        print('epochs', '\t\t', 'validation err', '\t\t', 'gradnorm', '\t\t', 'average training sample err')
        
        # starting point
        if starting_x != None:
            x = starting_x
            self.model.set_parameters(self.manifol2params(x))
        else:    
            x = self.params2manifold(self.model.get_parameters())
            
        x_opt = x
        
        # initializing errors
        self.training_err = self.cost(x)    
        err_sum = self.training_err
        err_strip = np.zeros(5)
        k = 0
        err_strip[k] = self.training_err
        k = k + 1
        self.validation_err = self.error(self.validation_set)
        cost_list.append([0,self.validation_err])
        self.opt_val_err = self.validation_err
        
        # first step
        t = 1
        
        # calculating riemannian gradient
        egrad = self.egrad(x)
        grad = self.man.rgrad(x, egrad)
        
        # 1st and 2nd moments estimates
        m = [(1-b1)*grad[0],(1-b1)*grad[1]]
        v = [(1-b2)*grad[0]**2,(1-b2)*grad[1]**2]

        v_bar = v
        
        # backuping x for future parallel transport
        x_bak = x

        # AMSGrad update rule
        dir = [-a*m[0]/(np.sqrt(v_bar[0])+eps),-a*m[1]/(np.sqrt(v_bar[1])+eps)]

        # retraction
        x = self.man.retr(x,dir)
        
        # parallel tranport of m and v to new manifold point
        m = self.man.transp(x_bak,x,m)
        v = self.man.transp(x_bak,x,v)

        # changing model's parameteres
        self.model.set_parameters(self.manifold2params(x))

        max_epochs = 0
        
        # repeat till early stopping
        while t<1000 :
            # step t
            t = t + 1
            max_epochs = max_epochs + 1
            
            # generating new minibatch
            self.batch = self.batch_gen()

            # updating errors
            cost = self.cost(x)
            err_sum = err_sum + cost
            self.training_err = err_sum/t
            err_strip[k] = cost
            k = k+1
            if t%5 == 0 :
                self.validation_err = self.error(self.validation_set)
                cost_list.append([t,self.validation_err])
                k = 0
                if self.validation_err < self.opt_val_err :
                    self.opt_val_err = self.validation_err
                    x_opt = x
                    max_epochs = 0
                #print(t, '\t\t', self.validation_err, '\t\t', gradnorm, '\t\t', self.training_err)
                #if self.GL()/self.Pk(err_strip, 5) > 0.1 :
                    #break
                #if max_epochs > 200 :
                    #print('Maximum number of epochs reached')
                    #break


            # new gradient
            egrad = self.egrad(x)
            grad = self.man.rgrad(x, egrad)
            gradnorm = self.man.norm(x, grad)
            
            # 1st and 2nd moments estimates
            m[0] = b1*m[0] + (1-b1)*grad[0]
            m[1] = b1*m[1] + (1-b1)*grad[1]
            
            v[0] = b2*v[0] + (1-b2)*grad[0]**2
            v[1] = b2*v[1] + (1-b2)*grad[1]**2

            v_bar = [np.maximum(v[0],v_bar[0]), np.maximum(v[1],v_bar[1])]
            
            # backuping x for future parallel transport            
            x_bak = x

            # adam update rule
            dir = [-a*m[0]/(np.sqrt(t)*(np.sqrt(v_bar[0])+eps)),-a*m[1]/(np.sqrt(v_bar[1])+eps)]
            
            # retraction
            x = self.man.retr(x,dir)
            
            # parallel tranport of m and v to new manifold point
            m = self.man.transp(x_bak,x,m)
            v = self.man.transp(x_bak,x,v)
            
            self.model.set_parameters(self.manifold2params(x))

        self.model.set_parameters(self.manifold2params(x_opt))
        
        return np.asarray(cost_list)

