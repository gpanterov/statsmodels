
import numpy as np
from scipy import integrate, stats
import KernelFunctions as kf
import scipy.optimize as opt

kernel_func=dict(wangryzin=kf.WangRyzin, aitchisonaitken=kf.AitchisonAitken,
                 epanechnikov=kf.Epanechnikov,gaussian=kf.Gaussian)


class LeaveOneOut(object):
    # Written by Skipper
    """
    Generator to give leave one out views on X

    Parameters
    ----------
    X : array-like
        Nd array

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
        N,K = np.shape(X)
        for i in xrange(N):
            index = np.ones([N,K], dtype=np.bool)
            index[i,:] = False
            yield X[index].reshape([N-1,K])


class UKDE(object):

    
    def __init__ (self,tdat,var_type,bw=False,bwmethod=False):
        self.tdat=tdat
        self.var_type=var_type
        self.all_vars,self.all_vars_type=self.GetAllVars()
        self.N,self.K=np.shape(self.all_vars)
        self.bw_func=dict(normal_reference=self.normal_reference,cv_ml=self.cv_ml)
        self.bw=self.get_bw(bw,bwmethod)
        

        
    def GetAllVars(self):
        for var in self.tdat:
            var=np.asarray(var)
            var=var.reshape([len(var),1])
        return np.concatenate(self.tdat,axis=1),self.var_type

    
    def pdf(self, bw, tdat,edat,var_type):
        return GPKE(bw,tdat=tdat,edat=edat,var_type=var_type)/self.N
    
    def fit_pdf(self,edat=False):
        if edat==False: edat=self.all_vars
        return self.pdf(self.bw,self.all_vars,edat,self.all_vars_type)
    
    def loo_likelihood(self,bw):
        LOO=LeaveOneOut(self.all_vars)
        i=0
        L=0
        for X_j in LOO:
            f_i=self.pdf(bw=bw,tdat=-X_j,edat=-self.all_vars[i,:],var_type=self.all_vars_type)/(self.N-1)           
            i+=1
            L+=np.log(f_i)       
        return -L

    def cv_ml(self):
        h0=self.normal_reference()
        bw=opt.fmin(self.loo_likelihood,x0=h0,maxiter=1e3,maxfun=1e3,disp=0)
        return bw
        
    def normal_reference(self):
        c=1.06        
        X=np.std(self.all_vars,axis=0)       
        return c*X*self.N**(-1./(4+self.K))

    def get_bw(self,bw,bwmethod):
        assert (bw!=False or bwmethod!=False) # either bw or bwmethod should be input by the user
        
        if bw!=False:
            return np.asarray(bw)
        if bwmethod!=False:
            self.bwmethod=bwmethod
            bwfunc=self.bw_func[bwmethod]
            return bwfunc()

class CKDE(UKDE):
    def __init__(self,tydat,txdat,dep_type,indep_type,bw=False,bwmethod=False):
        self.tydat=tydat
        self.txdat=txdat
        self.dep_type=dep_type
        self.indep_type=indep_type
        self.all_vars,self.all_vars_type=self.GetAllVars()

    def GetAllVars(self):
        for var in self.tydat:
            var=np.asarray(var)
            var=var.reshape([len(var),1])
        for var in self.txdat:
            var=np.asarray(var)
            var=var.reshape([len(var),1])
        av=np.concatenate((self.tydat,self.txdat),axis=1)
        avt=self.dep_type+self.indep_type
        return av,avt
    def pdf(self,bw,tdat,edat,var_types):
        GPKE(bw,tdat=tdat,edat=edat,var_type=var_type)/GPKE(bw[self.K_dep::],tdat=tdat[:,self.K_dep::],edat=edat[:,self.K_dep::],var_type=var_type)
class unconditional_bw():
    def __init__(self,tdat,var_type,bw=False,bwmethod=False):
        for var in tdat:
            var=np.asarray(var)
            var=var.reshape([len(var),1])
        
        self.tdat=np.concatenate(tdat,axis=1)
        self.N,self.K=np.shape(self.tdat)
        self.var_type=var_type
##        self.var_type_list=np.asarray(list(var_type))
##        self.iscontinuous=np.where(self.var_type_list=='c')[0]
##        self.isordered=np.where(self.var_type_list=='o')[0]
##        self.isunordered=np.where(self.var_type_list=='u')[0]
        self.bw_func=dict(normal_reference=self.normal_reference,cv_ml=self.cv_ml)
        
        self.bw=self.get_bw(bw,bwmethod)
        
        
    def get_bw(self,bw,bwmethod):
        assert (bw!=False or bwmethod!=False) # either bw or bwmethod should be input by the user
        
        if bw!=False:
            return np.asarray(bw)
        if bwmethod!=False:
            self.bwmethod=bwmethod
            bwfunc=self.bw_func[bwmethod]
            return bwfunc()
        
    def normal_reference(self):
        c=1.06        
        X=np.std(self.tdat,axis=0)       
        return c*X*self.N**(-1./(4+self.K))
    
    def cv_ml (self):
        h0=self.normal_reference()
        bw=opt.fmin(self.loo_likelihood, x0=h0,maxiter=1e3,maxfun=1e3,disp=0)
        return bw

    def loo_likelihood(self,bw):
        LOO=LeaveOneOut(self.tdat)
        i=0
        L=0
        for X_j in LOO:
            f_i=GPKE(bw,tdat=-X_j,edat=-self.tdat[i,:],var_type=self.var_type)/(self.N-1)           
            i+=1
            L+=np.log(f_i)       
        return -L
    
    def pdf(self,edat=False):
        if edat==False: edat=self.tdat
        return GPKE(self.bw,tdat=self.tdat,edat=edat,var_type=self.var_type)/self.N


class conditional_bw(object):
    def __init__ (self, tydat,txdat,dep_type,indep_type,bw=False,bwmethod=False):
        for var in tydat:
            var=np.asarray(var)
            var=var.reshape([len(var),1])
        for var in txdat:
            var=np.asarray(var)
            var=var.reshape([len(var),1])
            
        self.tydat=np.concatenate(tydat,axis=1)
        self.txdat=np.concatenate(txdat,axis=1)
        self.N,self.K_dep=np.shape(self.tydat)
        self.K_indep=np.shape(self.txdat)[1]
        
        self.dep_type=dep_type; self.indep_type=indep_type
        self.bw_func=dict(normal_reference=self.normal_reference,cv_ml=self.cv_ml)
        
        self.bw=self.get_bw(bw,bwmethod)
        
    def get_bw(self,bw,bwmethod):
        assert (bw!=False or bwmethod!=False) # either bw or bwmethod should be input by the user
        
        if bw!=False:
            return np.asarray(bw)
        if bwmethod!=False:
            self.bwmethod=bwmethod
            bwfunc=self.bw_func[bwmethod]
            
            return bwfunc()
        
    def normal_reference(self):
        c=1.06
        yx=np.concatenate((self.tydat,self.txdat),axis=1)
        Y=np.std(yx,axis=0); #X=np.std(self.txdat,axis=0)
         
        return c*Y*self.N**(-1./(4+self.K_dep+self.K_indep))#,c*X*self.N**(-1./(4+self.K_indep))

    def cv_ml (self):
        h0=self.normal_reference()
        
        bw=opt.fmin(self.loo_likelihood, x0=h0,maxiter=1e3,maxfun=1e3,disp=0)
        return bw

    def loo_likelihood(self,bw):
        #bw=np.reshape(bw,(self.K_dep+self.K_indep,))
        data=np.concatenate((self.tydat,self.txdat),axis=1)
        yLOO=LeaveOneOut(data)
        xLOO=LeaveOneOut(self.txdat).__iter__()
        i=0
        L=0
        for Y_j in yLOO:
            X_j=xLOO.next()
            f_yx=GPKE(bw,tdat=-Y_j,edat=-data[i,:],var_type=(self.dep_type+self.indep_type))
            f_x=GPKE(bw[self.K_dep::],tdat=-X_j,edat=-self.txdat[i,:],var_type=self.indep_type)
            f_i=f_yx/f_x
            i+=1
            L+=np.log(f_i)       
        return -L
    def pdf(self,eydat=False,exdat=False):
        if eydat==False: eydat=np.concatenate((self.tydat,self.txdat),axis=1)
        if exdat==False: exdat=self.txdat
        bw=self.bw
        f_yx=GPKE(bw,tdat=np.concatenate((self.tydat,self.txdat),axis=1),edat=eydat,var_type=(self.dep_type+self.indep_type))
        f_x=GPKE(bw[self.K_dep::],tdat=self.txdat,edat=exdat,var_type=self.indep_type)
        return (f_yx/f_x)
        
def GPKE(bw,tdat,edat,var_type,ckertype='gaussian',okertype='wangryzin',ukertype='aitchisonaitken'):
    
    var_type=np.asarray(list(var_type))
    iscontinuous=np.where(var_type=='c')[0]
    isordered=np.where(var_type=='o')[0]
    isunordered=np.where(var_type=='u')[0]
    
    if tdat.ndim>1:
        N,K=np.shape(tdat)
    else:
        K=1
        N = np.shape(tdat)[0]
        tdat=tdat.reshape([N,K])
    
    if edat.ndim>1:
        N_edat=np.shape(edat)[0]
    else:
        N_edat=1
        edat=edat.reshape([N_edat,K])
    
    bw=np.reshape(np.asarray(bw),(K,))  #must remain 1-D for indexing to work
    dens=np.empty([N_edat,1])
       
    for i in xrange(N_edat):
        
        Kval=np.concatenate((
        kernel_func[ckertype](bw[iscontinuous],tdat[:,iscontinuous],edat[i,iscontinuous]),
        kernel_func[okertype](bw[isordered],tdat[:,isordered],edat[i,isordered]),
        kernel_func[ukertype](bw[isunordered],tdat[:,isunordered],edat[i,isunordered])
        ),axis=1)
        
        dens[i]=np.sum(np.prod(Kval,axis=1))*1./(np.prod(bw[iscontinuous]))
    return dens
    
def likelihood_cv(bw,data,var_type):
    """
    Returns the leave one out log likelihood

    Parameters
    ----------
    h : float
        The bandwdith paremteter value (smoothing parameter)
    x : arraylike
        The training data
    var_type : str
        Defines the type of variable. Can take continuous, ordered or unordered

    Returns
    -------
    L : float
        The (negative of the) log likelihood value

    References
    ---------
    Nonparametric econometrics : theory and practice / Qi Li and Jeffrey Scott Racine.
     (p.16)
    """

    #TODO: Extend this to handle the categorical kernels
    
    if data.ndim==2:
        N,K=np.shape(data)
    else:
        K=1
        N=len(data)
        data=data.reshape([N,K])
    LOO=LeaveOneOut(data)
    i=0
    L=0

    for X_j in LOO:
        f_i=GPKE(bw,tdat=-X_j,edat=-data[i,:],var_type=var_type)/(N-1)
        #if f_i==0: f_i+=1e-10
        i+=1
##        if f_i==0:
##            print "f_i is 0"
        L+=np.log(f_i)
    
    return -L

def bw_normal_ref(data):
    c=1.06
    if data.ndim==2:
        N,K=np.shape(data)
    else:
        K=1
        N=len(data)
        data=data.reshape([N,K])
    
    X=np.std(data,axis=0)
    
    return c*X*N**(-1./(4+K))


def bw_likelihood_cv(x,var_type,h0=None):
    """
    Returns the bandwidth parameter which maximizes the leave one out likelihood
    """
    
    if h0==None:h0=bw_normal_ref(x)
    bw = opt.fmin(likelihood_cv, x0=h0,args=(x,var_type),maxiter=1e3,maxfun=1e3)
    #bw=opt.brute(likelihood_cv,ranges=((0.,2.)),args=(x,var_type))
    #bw=opt.anneal(likelihood_cv,x0=h0,args=(x,var_type))
    return bw
                
##class udens_bw(object):
##    def __init__(self, data,isordered,isunordered,bwmethod):
##        self.data=data
##        self.N,self.K=np.shape(self.endog)
##        self.isordered=isordered
##        self.isunordered=isunordered
##        self.iscontinuous=np.delete(range(self.K),isordered+isunordered)
##        
##    def loo_log_kde(self,bw): # The Leave one out density estimator
##        LOO=LeaveOneOut(self.data)
##        i=0
##        L=0
##        for X_j in LOO:
##            f_i=GPKE(bw,endog=-X_j,exog=-X[i,:],isordered=self.isordered,isunordered=self.isunordered)
##            i+=1
##            L+=np.log(f_i)
##        return -L
##    def ml_cv(self, bw0=np.ones(self,K,dtype=float)):
##        bw = fmin(self.loo_log_kde,bw0)
##        self.bw=bw
##        return bw
    
