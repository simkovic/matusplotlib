import numpy as np
from scipy import stats

__all__=['lognorm','gamma','weibull']
def latinSquare(N=4):
    U=np.zeros((2**(N),N),dtype=int)
    for i in range(N):
        U[:,i]= np.mod(np.arange(U.shape[0])/2**(N-1-i),2)
    return U

def lognorm(mu=1,sigma=1,phi=0):
    ''' Y ~ log(X)+phi
        X ~ Normal(mu,sigma)
        mu - mean of X
        sigma - standard deviation of X
    '''
    return stats.lognorm(sigma,loc=-phi,scale=np.exp(mu))
    

def gamma(mu=1,sigma=1,phi=0):
    ''' Gamma parametrized by mean mu and standard deviation sigma'''
    return stats.gamma(a=np.power(mu/sigma,2),scale=np.power(sigma,2)/mu,loc=-phi)

def weibull(scale=1,shape=1,loc=0):
    '''  pdf =shape* (x/scale+loc)**(shape-1)
        * exp(-(x/scale+loc)**shape)
    '''
    return stats.weibull_min(shape,scale=scale,loc=-loc)
