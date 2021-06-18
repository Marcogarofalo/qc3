#from scipy.special import sph_harm # no longer used
import math
import numpy as np
import sums_mov as sums
import F2_mov as F2
from defns import y2, y2real, list_nnk, list_nnk_nnP, lm_idx, chop, full_matrix, shell_nnk_list,shell_list
from numba import jit,njit

@jit(nopython=True,fastmath=True) #FRL, this speeds up like 5-10%
def npsqrt(x):
    return np.sqrt(x)
@jit(nopython=True,fastmath=True) #FRL, this speeds up like 5-10%
def square(x):
    return x**2

@jit(nopython=True,parallel=True,fastmath=True) #FRL this speeds up scalar prod of two vectors. DONT use for matrices.
def mydot(x,y):
    res = 0.
    for i in range(3):
        res+=x[i]*y[i]
    return res


def ktinvroot(e,L,nk):
    k = nk*2*math.pi/L
    omk = npsqrt(1+k**2)
    E2k = npsqrt(e**2 + 1 - 2*e*omk)
    return 1/npsqrt(32.*math.pi*omk*E2k )

def rho1mH(e, L, nk):
    k = nk*2*math.pi/L;
    hhk = sums.hh(e, k);
#    print(1-sums.E2a2(e,k))

    if hhk<1:
        return npsqrt(1-sums.E2a2(e,k))*(1-hhk)
    else:
        return 0.

# Calculate Gtilde = G/(2*omega)
#@njit(fastmath=True,cache=True)
@njit(fastmath=True,cache=True)
def G(e, L, nnp, nnk,l1,m1,l2,m2,nnP):
    p = sums.norm(nnp) * 2. *math.pi/L
    k = sums.norm(nnk) * 2. *math.pi/L
    pk = sums.norm(np.add(nnk,nnp)) * 2. *math.pi/L
    nPk = sums.norm(nnk-nnP)
    Pk = nPk*2*math.pi/L
    nPp = sums.norm(nnp-nnP)
    Pp = nPp*2*math.pi/L

    omp = npsqrt(1+square(p))
    omk = npsqrt(1+square(k))
    #ompk = np.sqrt(1+pk**2)

    bkp2 = (e-omp-omk)**2 - (2*math.pi/L)**2*sums.norm(np.add(nnP ,-1*np.add(nnk,nnp)))**2

    out = sums.hh(e,p,Pp)*sums.hh(e,k,Pk)/(L**3 * 4*omp*omk*(bkp2-1))
    return out.real

# Just compute l'=l=0 portion
#@jit(fastmath=True,cache=True)
def Gmat00_nnP(E,L,nnP):
  nnk_list = list_nnk_nnP(E,L,nnP)
  N = len(nnk_list)
#  print(nnk_list)
#  print(list(nnk_list[0]))

  Gfull = np.zeros((N,N))
  for p in range(N):
#    nnp = list(nnk_list[p])
    nnp = nnk_list[p]
    for k in range(N):
#      nnk = list(nnk_list[k])
      nnk = nnk_list[k]
      Gfull[p,k] = G(E,L,np.array(nnp),np.array(nnk),0,0,0,0,nnP)
#      print(nnk, nnp, Gfull[p,k])

  return chop(Gfull)




# Just compute l'=l=0 portion
#@jit(fastmath=True,cache=True)
#def Gmat00(E,L):
#  nnk_list = shell_list(E,L)
#  N = len(nnk_list)
#  Gfull = np.zeros((N,N))
#  for p in range(N):
#    nnp = nnk_list[p]
#    for k in range(N):
#      nnk = nnk_list[k]
#      Gfull[p,k] = 0.
#      allnnk = shell_nnk_list(nnk)
#      for nnk_prime in allnnk:
#          Gfull[p,k] += G(E,L,np.array(nnp),np.array(nnk_prime),0,0,0,0) * len( shell_nnk_list(nnp)  )
#  return chop(Gfull)
