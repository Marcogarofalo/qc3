import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
from F2_mov import  Fmat00
from H_mat import Hmat00
from defns import chop,truncate
import projections as proj
from numba import jit,njit
import G_mov, K2i_mat
##############################################################
# Compute full matrix F3 = 1/L**3 * (Ft/3 - Ft@Hi@Ft)
# Uses new structure w/ faster Fmat, Hmat
##############################################################

#@jit(fastmath=True,cache=True)
def F3mat00(E,L,alpha,nnP,kcot,IPV=0):
  # F00 = truncate(Fmat00(E,L,alpha,nnP,IPV))
  # Gt00 = truncate(G_mov.Gmat00_nnP(E,L,nnP))
  # K2it00 = truncate(K2i_mat.K2inv_mat00_nnP(E,L,nnP,kcot,IPV))
  F00 = Fmat00(E,L,alpha,nnP,IPV)   # TB: changed list_nnk_nnP so that truncate is unnecessary
  Gt00 = G_mov.Gmat00_nnP(E,L,nnP)
  K2it00 = K2i_mat.K2inv_mat00_nnP(E,L,nnP,kcot,IPV)

  Hi00 = chop(LA.inv( K2it00 + F00 + Gt00  ))
  return 1/L**3 * chop((1/3*F00 - F00@Hi00@F00))
#  return Hi00

@njit(fastmath=True,cache=True)
def LAinv(x):
  return LA.inv(x)


def isoprojector(E,L,nnP):
  nnk_vec = list_nnk_nnP(E,L,nnP)


@jit(fastmath=True,cache=True)
def F3mat00iso(E,L,Lista0,r0,P0,a2,alpha,IPV=0):
  F00 = Fmat00(E,L,alpha,IPV)
  F00o3 = 1./3*F00
  Gt00 = Gmatrix.Gmat00(E,L)
  res = []
  ones = np.ones(len(F00))
  for a0 in Lista0:
    K2it00 = K2i_mat.K2inv_mat00(E,L,a0,r0,P0,IPV)
    Hi00 = chop(LAinv( K2it00 + F00 + Gt00  ))
    f3mat = 1/L**3 * chop((F00o3 - F00@Hi00@F00))
    res.append(1./(ones@f3mat@ones))
  return res
