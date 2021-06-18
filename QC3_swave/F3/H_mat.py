import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg

import F2_mov, G_mov, sums_mov as sums, K2i_mat
from defns import chop, full_matrix, truncate
from numba import jit
#################################################################
# Compute full matrix H = 1/(2*omega*K2) + Ftilde + Gtilde
#################################################################

# Just compute l'=l=0 portion of H
@jit(fastmath=True,cache=True)
def Hmat00(E,L,kcot,alpha,nnP,IPV=0):
  Ft00 = F2_alt.Fmat00(E,L,alpha,nnP,IPV)
  Gt00 = Gmatrix.Gmat00_nnP(E,L,nnP)
  K2it00 = K2i_mat.K2inv_mat00_nnP(E,L,nnP,kcot,IPV)
  H00 = chop(K2it00 + Ft00 + Gt00)
  return truncate(H00)
