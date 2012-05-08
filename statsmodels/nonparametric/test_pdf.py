

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import numpy as np
import csv
import nonparametric2 as nparam
import KernelFunctions as kf




NP=importr('np')
r=robjects.r

N=30
u=np.random.binomial(2,0.7,size=(N,1))
c1=np.random.normal(size=(N,1))
c2=np.random.normal(5,1,size=(N,1))
c3=np.random.normal(3,2,size=(N,1))


dens=nparam.conditional_bw(tydat=[c1,u],txdat=[c2], dep_type='cu',indep_type='c',bwmethod='normal_reference')
#print dens.bw
#dens1=nparam.unconditional_bw(tdat=[c1,u],var_type='cu',bwmethod='normal_reference')
#dens2=nparam.generic_kde(tdat=[wage,lwage],var_type='cc',bwmethod='normal_reference')
##print dens1.bw
##print dens2.bw
#dens2.fit_pdf()

##xy=dens1.pdf()
##dens2=nparam.unconditional_bw(tdat=[lwage],var_type='c',bw=[0.4])
##x=dens2.pdf()
##print xy/x, dens.pdf()

D={"S1": robjects.FloatVector(c1),"S2":robjects.FloatVector(c2),"S3":robjects.IntVector(u)}
df=robjects.DataFrame(D)
formula=r('S1+S3~S2')
r_bw=NP.npcdensbw(formula, data=df, bwmethod='normal-reference')  #obtain R's estimate of the
##print r_bw[1],r_bw[0]
##print r_bw[1],r_bw[0]



print "------------------------"*4
print 'the estimate by R is: ', r_bw[1],r_bw[0], '||||||', 'the estimate by SM is: ', dens.bw

print dens.pdf()
##print "------------------------"*4
###err += abs((sm_bw-r_bw[0][0])/sm_bw)  
##
###y_ord = np.random.binomial(n=3,p=0.3,size=100)
###sm_bw_unord = bandwidths.bw_likelihood_cv(y_ord,'ordered',c=3)
