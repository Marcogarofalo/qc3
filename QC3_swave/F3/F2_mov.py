import math
import numpy as np
#sqrt=np.sqrt;
pi=np.pi
from itertools import permutations as perms
from scipy.linalg import block_diag

import sums_mov as sums
from defns import list_nnk_nnP, shell_list_nnP, shell_nnk_list, perms_list, chop, full_matrix
from group_theory_defns import Dmat
from projections import l2_proj

from numba import jit,njit

@jit(nopython=True,fastmath=True,cache=True)
def sqrt(x):
    return np.sqrt(x)

@jit(nopython=True,fastmath=True,cache=True)
def myabs(x):
    return abs(x)



#@jit(fastmath=True,cache=True)
def Fmat00(E,L,alpha,nnP,IPV=0):
#  shells = shell_list_nnP(E,L,nnP)
  nnk_list = list_nnk_nnP(E,L,nnP)
  #print(len(nnk_list))
#  print(nnk_list)
  F_list = []
  for nnk in nnk_list:
    nk = sums.norm(nnk)
    Pk = sums.norm(np.array(nnk)-np.array(nnP))*2*math.pi/L   # TB: added np.array()
    k = nk*2*math.pi/L
    hhk = sums.hh(E, k,Pk)
    omk = sqrt(1. + k**2)

    F_list += [sums.F2KSS(E,L,np.array(nnk),0,0,0,0,alpha,np.array(nnP)) + hhk*IPV/(32*math.pi*2*omk)]
    #print(F_list)

#  print(F_list)
  return np.diag(F_list)




# Just compute l'=l=0 portion
#@jit(fastmath=True,cache=True)
#def Gmat00(E,L):
#  nnk_list = list_nnk(E,L)
#  N = len(nnk_list)
#  print(nnk_list)
#  print(list(nnk_list[0]))
#  Gfull = np.zeros((N,N))
#  for p in range(N):
#    nnp = list(nnk_list[p])
#    nnp = nnk_list[p]
#    for k in range(N):
#      nnk = list(nnk_list[k])
#      nnk = nnk_list[k]
#      Gfull[p,k] = G(E,L,np.array(nnp),np.array(nnk),0,0,0,0)
#  return chop(Gfull)
