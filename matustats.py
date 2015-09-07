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

def pcaEIG(A,highdim=None):
    """ performs principal components analysis 
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to features/attributes. 

    Returns :  
    coeff :
    is a p-by-p matrix, each column contains coefficients 
    for one principal component.
    score : 
    the principal component scores ie the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
    latent : 
    a vector containing the normalized eigenvalues (percent variance explained)
    of the covariance matrix of A.
    Reference: Bishop, C. (2006) PRML, Chap. 12.1
    """
    A=np.array(A)
    n=A.shape[0];m=A.shape[1]
    highdim = n<m
    assert n!=m
    M = (A-A.mean(1)[:,np.newaxis]) # mean-center data
    if highdim:
        [latent,coeff] = np.linalg.eigh(np.cov(M))
        coeff=M.T.dot(coeff)
        denom=np.sqrt((A.shape[1]-1)*latent[np.newaxis,:])
        coeff/=denom #make unit vector length
    else:
        [latent,coeff] = np.linalg.eigh(np.cov(M.T))
    score = M.dot(coeff)
    latent/=latent.sum()
    # sort the data
    indx=np.argsort(latent)[::-1]
    latent=latent[indx]
    coeff=coeff[:,indx]
    score=score[:,indx]
    assert np.allclose(np.linalg.norm(coeff,axis=0),1)
    return coeff,score,latent

def pcaNIPALS(K=5,tol=1e-4,verbose=False):
    ''' Reference:
            Section 2.2 in Andrecut, M. (2009).
            Parallel GPU implementation of iterative PCA algorithms.
            Journal of Computational Biology, 16(11), 1593-1599.
        TODO - replace custom linear algebra (e.g. XmeanCenter) with
        numpy algebra
            
    '''
    if verbose: print 'Mean centering columns'
    XmeanCenter(1)
    latent=[]
    for k in range(K):
        lam0=0;lam1=np.inf
        T=np.matrix(XgetColumn(k))
        if verbose: print 'Computing PC ',k
        h=0
        while abs(lam1-lam0)>tol and h<100:
            P=Xleftmult(T,True)
            P=P/np.linalg.norm(P)
            T=Xleftmult(P)
            lam0=lam1
            lam1=np.linalg.norm(T)
            if verbose: print '\t Iteration '+str(h)+', Convergence =', abs(lam1-lam0)
            h+=1
        latent.append(lam1)
        XminusOuterProduct(T,P)
        #np.save(inpath+'T%02d'%k,T)
        np.save(inpath+'coeffT%d'%k,P.T)
    np.save(inpath+'latent',latent)
