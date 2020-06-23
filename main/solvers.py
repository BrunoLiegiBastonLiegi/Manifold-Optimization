import numpy as np
import scipy as sp
from abc import ABC, abstractmethod
import math
from scipy import optimize
from manifold import Manifold
from theta import rtbm
import time

 


class Solver(ABC):
    """Abstract solver class for manifold optimization.

    """
    
    def __init__(self, cost, model, x_data, y_data=None, verbosity=2, batchsize=256, validation=0.33):

        assert (type(model) == rtbm.RTBM), "works only with RTBM objects"

        self.input_cost = cost
        self.model = model
        self.x_data = x_data
        self.y_data = y_data
        self.batchsize = batchsize
        self.validation_set, self.training_set = self.set_init(validation)
        self.input_cost.__init__(self.training_set)
        self.val_cost = self.input_cost.__class__(self.validation_set)
        #self.batch = self.batch_gen()
        self.batch = self.training_set
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
        val = np.asarray(val)
        train = np.asarray(train)
        return val.reshape(1,val.size), train.reshape(1,train.size)      
    
    def batch_gen(self):
        index = [int(np.random.rand()*self.training_set.size) for j in range(self.batchsize)]
        return np.asarray([self.training_set[0][k] for k in index]).reshape(1,self.batchsize)

    def error(self, set):
        return self.input_cost.cost(self.model(set.reshape(1,set.size)))   #warning many calls to reshape!!!
    
    def GL(self):  # generalization loss in %
        return 100*(self.validation_err/self.opt_val_err - 1)

    def Pk(self, strip, k):  # learning progress on the strip of dim k
        return 1000*(np.sum(strip)/(k*np.amin(strip)) - 1)

    def flat_cost(self, params):        
        self.model.set_parameters(params)
        return self.input_cost.cost(self.model(self.input_cost.get_bins()))

    def cost(self, X):
        bak = self.model.get_parameters()
        self.model.set_parameters(self.manifold2params(X))
        #return self.flat_cost(params)
        cost = self.input_cost.cost(self.model(self.input_cost.get_bins()))
        self.model.set_parameters(bak)
        return cost
        

    """
    def egrad(self, X):
        print('optimize')
        start = time.time()
        g = optimize.approx_fprime(self.manifold2params(X), 
                                   self.flat_cost, 1e-8)
        end = time.time()
        print('Done in ', end - start)
        
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
    """
    def egrad(self, x): #warning does not use simmetry property
        f0 = self.cost(x)
        dx = 1e-8
        n = self.model._Nh + self.model._Nv
        egrad = [np.zeros(n),np.zeros((n,n))]
        ei = [np.zeros(n),np.zeros((n,n))]
        for k in range(n):
            ei[0][k] = dx
            df = self.cost(self.man.retr(x,ei)) - f0
            egrad[0][k] = df/dx
            ei[0][k] = 0
        for k in range(n):
            for j in range(n):
                ei[1][k,j] = dx
                df = self.cost(self.man.retr(x,ei)) - f0
                egrad[1][k,j] = df/dx
                ei[1][k,j] = 0
        return egrad
        
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
    """CMA evolution strategy on manifold.

    """
    
    def train(self, starting_x=None, popsize=4, m2=3):

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

            
        vc = 0
        vs = 0
        best_cost = float('Inf')
        best_x = x
        
        #while True:
        for i in range(3000):

            cost = self.val_cost.cost(self.model(self.val_cost.get_bins()))
            if i%5==0:
                print( i, '\t\t', cost,'\t\t')
            if cost < best_cost:
                best_cost = cost
                best_x = x
            # sampling new generation starting from x
            ##print('Sampling new gen')
            #start = time.time()
            self.new_gen(x)
            #end = time.time()
            #print('--> Done in ', end-start, 's')
            
            # performing recombination based on best fitted individuals
            v_best = self.recombination()

            # updating evolution paths
            vc = (1-self.cc)*vc + np.sqrt(self.cc*(2-self.cc)*self.meff)*v_best/self.s  # covariance path
            vs = np.real((1-self.cs)*vs + np.sqrt(self.cs*(2-self.cs)*self.meff)/self.s*np.dot(sp.linalg.sqrtm(sp.linalg.inv(self.C)),v_best))  # sigma path
            
            # update covariance matrix
            self.C_update(vc)
            
            # update stepsize
            self.s_update(x, vs)

            # backuping manifold point for future parallel transport
            x_bak = x
            
            # moving to new manifold point
            x = self.man.retr(x, self.params2manifold(v_best))
            self.model.set_parameters(self.manifold2params(x))
                       
            # parallel transport
            vc, vs = self.transport(x_bak, x, vc, vs)
            
        self.model.set_parameters(self.manifold2params(best_x))
        print('*** Best Solution: \n np.array(', self.model.get_parameters(), ')')
        
    def init(self):
        w = [np.log((self.m2+1)/i) for i in range(1,self.m2+1)]
        self.w = w / np.sum(w)
        self.meff = 1 / np.sum(self.w**2)
        self.N = int(np.sum(self.man.dim))
        self.cc = 4/(self.N+4)
        self.cs = (self.meff+2)/(self.N+self.meff+3)
        self.ccov = 2/(self.meff*(self.N+np.sqrt(2))**2) + (1-1/self.meff)*min(1,(2*self.meff-1)/((self.N+2)**2+self.meff))
        self.ds = 1 + 2*max(0,np.sqrt((self.meff-1)/(self.N+1))-1) + self.cs
        self.s = 0.1   # step size
        self.C = np.identity(self.N)
        self.pop = [[0,0] for i in range(self.popsize)]
        
    def new_gen(self, x):
        v = self.s*np.random.multivariate_normal(np.zeros(self.N), self.C, size=self.popsize)
        for i in range(self.popsize):
            self.pop[i][0] = v[i]
            #print('Retraction')
            #start = time.time()
            tmp = self.man.retr(x, self.params2manifold(v[i]))
            #print(tmp)
            #end = time.time()
            #print('--> Done in ', end-start, 's')
            #print('Cost')
            self.pop[i][1] = self.cost(tmp)
            #end = time.time()
            #print('--> Done in ', end-start, 's')
            while math.isnan(self.pop[i][1]) or self.pop[i][1] == float('Inf'):
                print('*****************punto problematico')
                v[i] = self.s*np.random.multivariate_normal(np.zeros(self.N), self.C, size=1)
                self.pop[i][0] = v[i]
                self.pop[i][1] = self.cost(self.man.retr(x, self.params2manifold(v[i])))
        self.pop.sort(key = lambda x: x[1])
    
    def recombination(self):
        best = 0
        for i in range(self.m2):
            best = best + self.w[i]*self.pop[i][0]
        return best

    def C_update(self, vc):
        tmp = 0
        for i in range(self.m2):
            v = np.asarray(self.pop[i][0]).reshape(self.N,1)
            tmp = tmp + self.w[i]*np.dot(v,v.T)
        self.C = self.ccov*(1-1/self.meff)*(1/self.s**2)*tmp + (1-self.ccov)*self.C + self.ccov/self.meff*np.dot(vc.reshape(self.N,1),vc.reshape(self.N,1).T)

    def s_update(self, x, vs):
        E = np.sqrt(2)*sp.special.gamma((self.N+1)/2)/sp.special.gamma(self.N/2)
        Vs = self.params2manifold(vs)
        self.s = min(self.s*np.exp((self.man.norm(x, Vs)/E-1)*self.cs/self.ds), 0.001)
    
    def transport(self, x1, x2, vc, vs): # check that is all right
        # first transport vc & vs
        Vc = self.man.transp(x1, x2, self.params2manifold(vc))
        Vs = self.man.transp(x1, x2, self.params2manifold(vs))
        # now transport C
        #e = self.coord_sys()
        #self.Ctransp(x1, x2, e)
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
        newC = np.zeros((self.N, self.N))
        for k in range(self.N):
            for l in range(self.N):
                tmp = 0
                for i in range(self.N):
                    for j in range(self.N):
                        tmp = tmp + et[k][i]*et[l][j]*self.C[i][j]
                newC[k][l] = tmp
        self.C = newC








        



class ADAM(Solver):
    """ADAM on manifold.

    """

    def train(self, starting_x=None):
        
        self.b1 = 0.994
        self.b2 = 0.999
        a = 0.01
        eps = 1e-8
        
        print('epochs', '\t\t', 'validation err', '\t\t', 'gradnorm', '\t\t')
        
        # starting point
        if type(starting_x) == np.ndarray:
            x = starting_x
            self.model.set_parameters(self.manifol2params(x))
        else:    
            x = self.params2manifold(self.model.get_parameters())
        
        best_x = x
        best_cost = float('Inf')

        t = 0
        v = [0,0]
        m = [0,0]
        iter_time = 0   

        gradnorm = float('Inf')
                
        # repeat till early stopping
        while gradnorm > 0.01 :
            
            start = time.time()            
            
            # step t
            t = t +1

            # updating errors
            if t%5 == 0 :
                cost = self.val_cost.cost(self.model(self.val_cost.get_bins()))
                if cost < best_cost:
                    best_cost = cost
                    best_x = x
                print(t, '\t\t', cost, '\t\t', self.man.norm(x, grad), '\t\t')
                
            # new gradient
            egrad = self.egrad(x)
            grad = self.man.rgrad(x, egrad)
            enorm = self.man.enorm(x, grad)
            pnorm = self.man.pnorm(x, grad)
            gradnorm = np.sqrt(enorm + pnorm)
            
            # 1st and 2nd moments estimates
            m[0] = self.b1*m[0] + (1-self.b1)*grad[0]
            m[1] = self.b1*m[1] + (1-self.b1)*grad[1]
            
            v[0] = self.b2*v[0] + (1-self.b2)*enorm
            v[1] = self.b2*v[1] + (1-self.b2)*pnorm

            #v_bar = [np.maximum(enorm,v_bar[0]),np.maximum(pnorm,v_bar[1])]
            v_bar = [v[0]/(1-self.b2t(t)), v[1]/(1-self.b2t(t))]
            m_bar = [m[0]/(1-self.b1t(t)), m[1]/(1-self.b1t(t))]
            
            # backuping x for future parallel transport            
            x_bak = x

            # update rule            
            dir = [-a*m_bar[0]/(np.sqrt(v_bar[0])+eps), -a*m_bar[1]/(np.sqrt(v_bar[1])+eps)]
            
            # retraction
            x = self.man.retr(x,dir)
            self.model.set_parameters(self.manifold2params(x))
            
            # parallel tranport of m and v to new manifold point
            m = self.man.transp(x_bak,x,m)
            #v = self.man.transp(x_bak,x,v)
            
            end = time.time()            
            iter_time = iter_time + end - start
            
        self.model.set_parameters(self.manifold2params(best_x))
        print('*** Best Solution: \n np.array(', self.model.get_parameters(), ')')
        
        return (best_cost, iter_time/t, t)

    def b1t(self, t):
        return self.b1/np.sqrt(t)
    
    def b2t(self, t):
        return self.b2/np.sqrt(t)


class SGD(Solver):
    """
    Stochastic Gradient Descent with momentum
    """

    def train(self, starting_x=None):

        self.m = 0.994
        self.lr = 0.001
        
        print('epochs', '\t\t', 'validation err', '\t\t', 'gradnorm', '\t\t')
        
        # starting point
        if type(starting_x) == np.ndarray:
            x = self.params2manifold(starting_x)
            self.model.set_parameters(self.manifold2params(x))
        else:    
            x = self.params2manifold(self.model.get_parameters())
            
        best_cost = float('Inf')
        best_x = x

        t = 0
        iter_time = 0
        v = [0,0]

        gradnorm = float('Inf')
        
        # repeat till early stopping
        while gradnorm > 0.01 :

            start = time.time()            
            
            # step t
            t = t + 1

            # updating errors
            if t%5 == 0 :
                cost = self.val_cost.cost(self.model(self.val_cost.get_bins()))
                if cost < best_cost:
                    best_cost = cost
                    best_x = x
                print(t, '\t\t', cost, '\t\t', self.man.norm(x, grad), '\t\t')

            # new gradient
            egrad = self.egrad(x)
            grad = self.man.rgrad(x, egrad)
            gradnorm = self.man.norm(x, grad)
            
            # dir update
            v = [self.m * v[0] + self.lr * grad[0], self.m * v[1] + self.lr * grad[1]]
            dir = [-v[0],-v[1]]
            
            # backuping x for future parallel transport            
            x_bak = x

            # retraction
            x = self.man.retr(x,dir)
            self.model.set_parameters(self.manifold2params(x))
            
            # parallel transport
            v = self.man.transp(x_bak,x,v)

            end = time.time()
            iter_time = iter_time + end - start

        self.model.set_parameters(self.manifold2params(best_x))
        print('*** Best Solution: \n np.array(', self.model.get_parameters(), ')')

        return (best_cost, iter_time/t, t)




class NAG(Solver):
    """
    Nesterov Accelerated Gradient
    """

    def train(self, starting_x=None):

        self.m = 0.9
        self.lr = 0.001
        
        print('epochs', '\t\t', 'validation err', '\t\t', 'gradnorm', '\t\t')
        
        # starting point
        if type(starting_x) == np.ndarray:
            x = self.params2manifold(starting_x)
            self.model.set_parameters(self.manifold2params(x))
        else:    
            x = self.params2manifold(self.model.get_parameters())
            
        best_cost = float('Inf')
        best_x = x

        t = 0
        iter_time = 0
        v = [0,0]

        gradnorm = float('Inf')
        
        # repeat till early stopping
        while gradnorm > 0.01 :

            start = time.time()
            
            # step t
            t = t + 1

            # updating errors
            if t%5 == 0 :
                cost = self.val_cost.cost(self.model(self.val_cost.get_bins()))
                if cost < best_cost:
                    best_cost = cost
                    best_x = x
                print(t, '\t\t', cost, '\t\t', self.man.norm(x, grad), '\t\t')

            # new gradient
            egrad = self.egrad(x)
            grad = self.man.rgrad(x, egrad)
            gradnorm = self.man.norm(x, grad)
            
            # dir update
            v = [self.m * v[0] + self.lr * grad[0], self.m * v[1] + self.lr * grad[1]]
            dir = [-self.m * v[0] -self.lr * grad[0], -self.m * v[1] -self.lr * grad[1]]
            
            # backuping x for future parallel transport            
            x_bak = x

            # retraction
            x = self.man.retr(x,dir)
            self.model.set_parameters(self.manifold2params(x))
            
            # parallel transport
            v = self.man.transp(x_bak,x,v)

            end = time.time()
            iter_time = iter_time + end - start

        self.model.set_parameters(self.manifold2params(best_x))
        print('*** Best Solution: \n np.array(', self.model.get_parameters(), ')')

        return (best_cost, iter_time/t, t)
