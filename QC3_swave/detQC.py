import pickle
from numba import jit, njit, prange
import sys
import os
from scipy import interpolate
from defns import chop
from scipy.optimize import fsolve
from F3 import sums_mov as sums
import time
import numpy as np
import scipy
import math
import defns
from F3 import G_mov as Gm
from F3 import K2i_mat as K2
from F3 import F3_mat as F3mat
from F3 import F2_mov as F2
from multiprocessing import Pool
import itertools
from sys import platform as sys_pf
import matplotlib
matplotlib.use("TkAgg")
# if sys_pf == 'darwin':
#    import matplotlib
#    matplotlib.use("TkAgg")
#    import os, sys, time, pickle
cwd = os.getcwd()
#import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0, cwd+'/F3')

# @njit(fastmath=True)


def kcot_2par(qk2, params):
    a0 = params[0]
    r0 = params[1]
    #print("calling kcot_2par",params[0],params[1])

    return 1/a0 + r0*qk2/2.0


# @njit(fastmath=True)
def kiso_2par(Delta, params):
    K0 = params[0]
    K1 = params[1]
    return K0 + K1*Delta

# @njit(fastmath=True)


def kiso_1par(Delta, params):
    K0 = params[0]
    return K0


def kiso_pole(Delta, params):
    ecm2 = Delta*9.0+9.0
    c = params[0]
    MR2 = params[1]
    r = -c/(ecm2-MR2)
    if len(params) > 2:
        r += params[2]
    #print("calling kiso_pole",params[0],params[1], ecm2)
    return r

def kiso_pole_fix(Delta, params):
    ecm2 = Delta*9.0+9.0
    c = params[0]
    MR2 = 9.6
    r = -c/(ecm2-MR2)
    #if len(params) > 1:
    r += params[1]
    #print("calling kiso_pole",params[0],params[1], ecm2)
    return r

def kiso_const(Delta, params):
    r = params[0]
    return r
def kiso_par(Delta, params):
    K0 = params[0]
    if len(params) > 1:
        K1 = params[1]
        K0+=K1
    return K0 

def QC3(e, L, nnP, kcot_in, params_kcot, kiso, params_kiso):
    
    #if( type(e)==type(np.ndarray([0])) ):        e=e[0]
    def kcot(qk2): return kcot_in(qk2, params_kcot)

    F00 = F3mat.F3mat00(e, L, 0.5, nnP, kcot)
    #ecm = np.sqrt(e**2 - (sums.norm(nnP)*2*math.pi/L)**2 )
    #Delta = (ecm**2 -9.0)/9.0
    ecm2 = (e**2 - (sums.norm(nnP)*2*math.pi/L)**2)
    Delta = (ecm2 - 9.0)/9.0

    Kiso = kiso(Delta, params_kiso)
    # print("Kiso=",Kiso)
    ones = np.ones(len(F00))
    Fiso = 1/(ones@F00@ones)
    #print("F+K(",e,")=",Fiso,Kiso)
    return Fiso + Kiso


def write_db_Fmat00():
    F3mat.write_db_Fmat00()


def find_sol(Estart, Eend, steps, L, nnP, kcot, params_kcot, kiso, params_kiso):
    #start = time.time()

    energies = np.linspace(Estart, Eend, steps)
    param = []
    for i in range(steps):
        param.append((energies[i], L, nnP, kcot,
                     params_kcot, kiso, params_kiso))
    #print('param', param[0])
    # print('E=',Estart,Eend)
    res = list(itertools.starmap(QC3, param))
    #print('Energies', energies)
    #print('QC3iso', res)
    func = interpolate.InterpolatedUnivariateSpline(energies, res)
    E3 = scipy.optimize.fsolve(func, 0.5*(Estart+Eend))
    #print('Estart=',Estart,'Eend=', Eend, 'Esol=',E3)
    #end = time.time()
    #print('python time:', end - start, ' s')
    write_db_Fmat00()
    return E3


# def find_sol(Estart, Eend, steps,L,nnP,kcot,params_kcot,kiso,params_kiso):

    #func=lambda e : QC3(e ,L,nnP,kcot,params_kcot ,kiso,params_kiso)

    ##print("   bracket=[ ",Estart," ,",Eend," ] -> [",func(Estart)," , ",func(Eend),"]" )
    # E3=scipy.optimize.root_scalar(func,method='brentq',bracket=[Estart,4], x0=0.5*(Estart+Eend), xtol=1e-2 )  #  bisect
    ##print(" QC3 converged= ",E3.converged , "in ",E3.iterations, " iteration" ,  "result =", E3.root )
    # if (not E3.converged):
    #print("error not coverged")

    # return E3.root


def print_find_sol(Estart, Eend, steps, L, nnP, kcot, params_kcot, kiso, params_kiso):
    print(Estart, Eend, steps, L, nnP, kcot, params_kcot, kiso, params_kiso)

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


def get_np_array(n1, n2, n3):
    return np.array([n1, n2, n3])


def find_2sol(Estart, deltaE, n, L, nnP, kcot, params_kcot, kiso, params_kiso):
    #tic1 = time.perf_counter()
    founds = 0
    def func(e): return QC3(e, L, nnP, kcot, params_kcot, kiso, params_kiso)
    # print("we are calling find_2sol n=",n," Lm=",L, "P=",params_kiso[0])
    #print(Estart, deltaE, n, L, nnP, kcot, params_kcot, kiso, params_kiso)
    E1=Estart
    f1 = func(E1)
    roots=[]
    while(founds < n+1):
        if (E1>3.2):
            print("decreasing step size to ", deltaE," sol founded:", founds,)
            E1=Estart
            #if (founds>0): 
            #    E1=roots[founds-1]
            roots=[]
            founds=0
            deltaE=deltaE/5.0

        f2 = func(E1+deltaE)   
        
        # print("iter =", iter, " F=",f1,f2, "   P=",params_kiso[0],  "E=",Estart)
        if (f1*f2 < 0):
            #physcal condition
            if (f1<0 and f2>0):
                #tic = time.perf_counter()
                sol = scipy.optimize.root_scalar(func, method='bisect', bracket=[
                                            E1, E1+deltaE], x0=(E1+0.5*deltaE), xtol=1e-8)  # bisect brentq            
                # if ((func(sol.root))>1e-6):
                #     print(sol)
                #     print("solution not foun  func(sol.root)=",func(sol.root),func(sol.root-1e-6),func(sol.root+1e-6)) 
                
            
                roots.append( sol.root)
                #toc = time.perf_counter()
                #print(f"brentq solver: {toc - tic:0.4f} seconds")
                #print("finding root in [", Estart,",",Estart+deltaE,"] n=",n , "   sol=",E3)
                founds+=1
            else :
                print("unphysical solution found")
        elif (f1 == 0):
            print("found exact root")
            n += 1
            roots.append( E1)
            founds+=1
        elif (f2 == 0):
            print("found exact root")
            n += 1
            roots.append( E1+deltaE)
            founds+=1
        
        E1 = E1+deltaE
        f1=f2
    #toc1 = time.perf_counter()
    #print(f"find_2sol: {toc1 - tic1:0.4f} seconds")
   
    write_db_Fmat00()
    return roots[n]


def energy3_alt(mom, L, model0, par0, Estart, parK, deltaEaux=2e-4):
    def kcot0(p2): return model0(p2, *par0)
    def faux(Ecm): return QC3(Ecm[0], L, mom, kcot0, parK)
    deltaE = deltaEaux
    E0, E1 = Estart, Estart+deltaE
    fE0, fE1 = faux([E0]), faux([E1])
    stop = 0
    listguess = []
    while(stop == 0):

        if(abs(E1-parK[1]) < 0.0002):
            deltaE = deltaEaux/100
            if(L > 5.0):
                deltaE = deltaEaux/50
        else:
            deltaE = deltaEaux

        E0 = E1
        E1 = E1 + deltaE

        fE0, fE1 = faux([E0]), faux([E1])
        if(fE0 < 0 and fE1 > 0):
            listguess.append(0.5*(E0+E1))
            if(L > 5.0):
                E1 = parK[1]-0.001
            if(len(listguess) == 2):
                stop = 1
        if(E0 > 3.20):
            stop = 2

    print(listguess)
    sols = []
    for guess in listguess:
        E3sol = fsolve(faux, guess)[0]
        sols.append(E3sol)
    if(stop == 2):
        sols = energy3_alt(mom, L, model0, par0, Estart,
                           parK, deltaEaux=deltaEaux/1.5)

    return sols
# exit()
