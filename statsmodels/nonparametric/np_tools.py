import numpy as np
import KernelFunctions as kf
import scipy.optimize as opt

kernel_func=dict(wangryzin = kf.WangRyzin, aitchisonaitken = kf.AitchisonAitken,
                 epanechnikov = kf.Epanechnikov, gaussian = kf.Gaussian)

Convolution_Kernels = dict (gaussian = kf.Gaussian_Convolution)

class LeaveOneOut(object):
    # Written by Skipper
    """
    Generator to give leave one out views on X

    Parameters
    ----------
    X : array-like
        2d array

    Examples
    --------
    >>> X = np.random.normal(0,1,[10,2])
    >>> loo = LeaveOneOut(X)
    >>> for x in loo:
    ...    print x

    Notes
    -----
    A little lighter weight than sklearn LOO. We don't need test index.
    Also passes views on X, not the index.
    """
    def __init__(self, X):
        self.X = np.asarray(X)

    def __iter__(self):
        X = self.X
        N, K = np.shape(X)

        for i in xrange(N):
            index = np.ones(N, dtype = np.bool)
            index[i] = False
            yield X[index, :]

def GPKE(bw, tdat, edat, var_type, ckertype = 'gaussian', okertype = 'wangryzin', ukertype = 'aitchisonaitken'):
    """
    Returns the non-normalized Generalized Product Kernel Estimator
    
    Parameters
    ----------
    bw: array-like
        The user-specified bandwdith parameters
    tdat: 1D or 2d array
        The training data
    edat: 1d array
        The evaluation points at which the kernel estimation is performed
    var_type: str
        The variable type (continuous, ordered, unordered)
    ckertype: str
        The kernel used for the continuous variables
    okertype: str
        The kernel used for the ordered discrete variables
    ukertype: str
        The kernel used for the unordered discrete variables
        
    """
    var_type = np.asarray(list(var_type))
    iscontinuous = np.where(var_type == 'c')[0]
    isordered = np.where(var_type == 'o')[0]
    isunordered = np.where(var_type == 'u')[0]
    
    if tdat.ndim > 1:
        N, K = np.shape(tdat)
    else:
        K = 1
        N = np.shape(tdat)[0]
        tdat = tdat.reshape([N, K])
    
    if edat.ndim > 1:
        N_edat = np.shape(edat)[0]
    else:
        N_edat = 1
        edat = edat.reshape([N_edat, K])
    
    bw = np.reshape(np.asarray(bw), (K,))  #must remain 1-D for indexing to work
    dens = np.empty([N_edat, 1])
       
    for i in xrange(N_edat):
        
        Kval = np.concatenate((
        kernel_func[ckertype](bw[iscontinuous], tdat[:, iscontinuous], edat[i, iscontinuous]),
        kernel_func[okertype](bw[isordered], tdat[:, isordered],edat[i, isordered]),
        kernel_func[ukertype](bw[isunordered], tdat[:, isunordered], edat[i, isunordered])
        ), axis=1)
        
        dens[i] = np.sum(np.prod(Kval, axis = 1))*1./(np.prod(bw[iscontinuous]))
    return dens


def Convolution_GPKE(bw, tdat, edat, var_type, ckertype = 'gaussian'):
    var_type = np.asarray(list(var_type))
    iscontinuous = np.where(var_type == 'c')[0]
    isordered = np.where(var_type == 'o')[0]
    isunordered = np.where(var_type == 'u')[0]
    
    if tdat.ndim > 1:
        N, K = np.shape(tdat)
    else:
        K = 1
        N = np.shape(tdat)[0]
        tdat = tdat.reshape([N, K])
    
    if edat.ndim > 1:
        N_edat = np.shape(edat)[0]
    else:
        N_edat = 1
        edat = edat.reshape([N_edat, K])
    
    bw = np.reshape(np.asarray(bw), (K,))  #must remain 1-D for indexing to work
    dens = np.empty([N_edat, 1])
       
    for i in xrange(N_edat):
        
        Kval = np.concatenate((
        Convolution_Kernels[ckertype](bw[iscontinuous], tdat[:, iscontinuous], edat[i, iscontinuous]),
#        Convolution_Kernels[okertype](bw[isordered], tdat[:, isordered],edat[i, isordered]),
#        Convolution_Kernels[ukertype](bw[isunordered], tdat[:, isunordered], edat[i, isunordered])
        ), axis=1)
        
        dens[i] = np.sum(np.prod(Kval, axis = 1))*1./(np.prod(bw[iscontinuous]))
    return dens


 
