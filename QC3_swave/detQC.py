from multiprocessing import Pool
import itertools  
from sys import platform as sys_pf
import matplotlib
matplotlib.use("TkAgg")
import os, sys, time, pickle
#if sys_pf == 'darwin':
#    import matplotlib
#    matplotlib.use("TkAgg")
#    import os, sys, time, pickle
cwd = os.getcwd()
import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
from F3 import F2_mov as F2
from F3 import F3_mat as F3mat
from F3 import K2i_mat as K2
from F3 import G_mov as Gm
import defns
import math
import scipy
import numpy as np
import time
from numba import jit,njit,prange
from F3 import sums_mov as sums
from scipy.optimize import fsolve
from defns import chop
from scipy import interpolate

@njit(fastmath=True)
def kcot_2par(qk2,params):
    a0 = params[0]
    r0 = params[1]
    return -1/a0 + r0*qk2
    
    
    

def QC3(e,L,nnP,kcot):
    F00 = F3mat.F3mat00(e,L,0.5,nnP,kcot)
    Kiso = 800
    #Kiso= const + c1* qk^2
#    K3 = 0*F00 + Kiso #useless
    ones = np.ones(len(F00))
    Fiso = 1/(ones@F00@ones)
    return Fiso  + Kiso


def find_sol(Estart, Eend, steps,L,nnP,kcot_in,params_kcot):
    kcot=lambda qk2 : kcot_in(qk2,params_kcot)
    energies = np.linspace(Estart, Eend,steps)
    param = []
    for i in range(steps):
        param.append((energies[i],L,nnP,kcot))
    res = list(itertools.starkcotmap(QC3, param))
    print('Energies', energies)
    print('QC3iso', res)
    func = interpolate.InterpolatedUnivariateSpline(energies, res)
    
    return scipy.optimize.fsolve(func, 0.5*(Estart+Eend) )


#L = 10.0
#nnP = np.array([0.,0.,0.])
#Estart = 3.0033
#Eend = 3.0040
#steps = 40

#start = time.time()
#sol= find_sol(Estart, Eend, steps,L,nnP,kcot_simple)  
#print('solution is:', 'E=',sol ,'Ecm=', np.sqrt(sol**2 - (sums.norm(nnP)*2*math.pi/L)**2))
#end = time.time()
#print('time is:', end - start, ' s')

#ecm = np.sqrt(e**2 - (sums.norm(nnP)*2*math.pi/L)**2 )
#delta = (ecm**2 -9.0)/9.0


def get_np_array(n1,n2,n3):
    return np.array([n1,n2,n3])

#exit()
