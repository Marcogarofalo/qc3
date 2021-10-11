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
#import os, sys, time, pickle
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

#@njit(fastmath=True)
def kcot_2par(qk2,params):
    a0 = params[0]
    r0 = params[1]
    #print("calling kcot_2par",params[0],params[1])

    return 1/a0 + r0*qk2/2.0
    
    
#@njit(fastmath=True)
def kiso_2par(Delta,params):
    K0 = params[0]
    K1 = params[1]
    return K0 + K1*Delta

#@njit(fastmath=True)
def kiso_1par(Delta,params):
    K0 = params[0]
    return K0  


def kiso_pole(Delta,params):
    ecm2=Delta*9.0+9.0
    
    c = params[0]
    MR2= params[1]
    #print("calling kiso_pole",params[0],params[1], ecm2)
    return -c/(ecm2-MR2)

def QC3(e, L, nnP, kcot_in, params_kcot, kiso, params_kiso ):

    kcot=lambda qk2 : kcot_in(qk2,params_kcot)

    F00 = F3mat.F3mat00(e,L,0.5,nnP,kcot)
    
    #ecm = np.sqrt(e**2 - (sums.norm(nnP)*2*math.pi/L)**2 )
    #Delta = (ecm**2 -9.0)/9.0
    ecm2 = (e**2 - (sums.norm(nnP)*2*math.pi/L)**2 )
    Delta = (ecm2**2 -9.0)/9.0
    
    Kiso = kiso(Delta, params_kiso)
    #print("Kiso=",Kiso)
    ones = np.ones(len(F00))
    Fiso = 1/(ones@F00@ones)
    return Fiso  + Kiso


def write_db_Fmat00():
    F3mat.write_db_Fmat00()

def find_sol(Estart, Eend, steps,L,nnP,kcot,params_kcot,kiso,params_kiso):
    #start = time.time()
    
    energies = np.linspace(Estart, Eend,steps)
    param = []
    for i in range(steps):
        param.append((energies[i],L,nnP,kcot,params_kcot ,kiso,params_kiso))
    #print('param', param[0])
    #print('E=',Estart,Eend)
    res = list(itertools.starmap(QC3, param))
    #print('Energies', energies)
    #print('QC3iso', res)
    func = interpolate.InterpolatedUnivariateSpline(energies, res)
    E3=scipy.optimize.fsolve(func, 0.5*(Estart+Eend) )
    #print('Estart=',Estart,'Eend=', Eend, 'Esol=',E3)
    #end = time.time()
    #print('python time:', end - start, ' s')
    write_db_Fmat00()
    return E3



#def find_sol(Estart, Eend, steps,L,nnP,kcot,params_kcot,kiso,params_kiso):
    
    #func=lambda e : QC3(e ,L,nnP,kcot,params_kcot ,kiso,params_kiso)
    
    ##print("   bracket=[ ",Estart," ,",Eend," ] -> [",func(Estart)," , ",func(Eend),"]" )
    #E3=scipy.optimize.root_scalar(func,method='brentq',bracket=[Estart,4], x0=0.5*(Estart+Eend), xtol=1e-2 )  #  bisect
    ##print(" QC3 converged= ",E3.converged , "in ",E3.iterations, " iteration" ,  "result =", E3.root )
    #if (not E3.converged):
        #print("error not coverged")
    
    #return E3.root


def print_find_sol(Estart, Eend, steps,L,nnP,kcot,params_kcot,kiso,params_kiso):
    print(Estart, Eend, steps,L,nnP,kcot,params_kcot,kiso,params_kiso)

#L = 28.0
#nnP = np.array([0.,0.,0.])
#Estart = 3.0033
#Eend = 3.0040
#steps = 40

#start = time.time()
#sol= find_sol(Estart, Eend, steps,L,nnP,kcot_2par, (1,0),kiso_2par, (1,0))  
#print('solution is:', 'E=',sol ,'Ecm=', np.sqrt(sol**2 - (sums.norm(nnP)*2*math.pi/L)**2))
#end = time.time()
#print('time is:', end - start, ' s')

#

def get_np_array(n1,n2,n3):
    return np.array([n1,n2,n3])


#exit()
