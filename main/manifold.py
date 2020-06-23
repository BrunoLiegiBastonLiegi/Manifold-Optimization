

import numpy as np
import scipy as sp




class Manifold(object):
    """Custom Manifold for an RTBM object"""

    def __init__(self, model):

        self.model = model

    @property
    def dim(self):
        nh = self.model._Nh 
        nv = self.model._Nv
        en = nh + nv
        pn = (nh*(nh+1) + nv*(nv+1))/2 + nh*nv 
        return (int(en),int(pn))
        
    def retr(self, x, v):
        er = x[0] + v[0]
        tmp = np.linalg.solve(x[1], v[1])
        pr = np.dot(x[1],sp.linalg.expm(tmp))
        return (er, pr)

    def pnorm(self, x, v):
        invx = sp.linalg.inv(x[1])
        tmp = np.dot(invx, np.dot(v[1], np.dot(invx, v[1])))
        return np.trace(tmp)       

    def enorm(self, x, v):
        return np.sum(v[0]**2)
    
    def norm(self, x, v):
        return np.sqrt(self.enorm(x,v) + self.pnorm(x,v))

    def transp (self, x1, x2, v):
        E = sp.linalg.sqrtm(np.dot(x2[1],np.linalg.inv(x1[1])))
        pt = np.dot(E,np.dot(v[1],E.T))
        return [v[0] ,pt]

    def diff(self, x1, x2):      # beware: it makes sense only in the tangent space
        return (x1[0]-x2[0], x1[1]-x2[1])

    def sum(self, x1, x2):      # beware: it makes sense only in the tangent space
        return (x1[0]+x2[0], x1[1]+x2[1])

    def rgrad(self, x, egrad):
        eg = egrad[0]
        pg = egrad[1]
        return (eg, 0.5*np.dot(x[1], np.dot(pg + pg.T, x[1])))

    

