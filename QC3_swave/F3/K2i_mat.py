import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
from scipy.linalg import block_diag
import sums_mov as sums
import defns
#from defns import omega, E2k, qst, list_nnk, lm_idx, full_matrix
from numba import jit,njit

@jit(nopython=True,fastmath=True,cache=True)
def E2k(e, a,Pk):
    return sqrt((e - sqrt(1+a**2))**2 - Pk**2)

@njit(fastmath=True,cache=True)
def qst2(E,k,Pk):
  Estar = E2k(E,k,Pk)
  if Estar >= 2:
    return  Estar**2/4 - 1  # can be real...                                                                                                                                             
  else:
    return -1*( 1-Estar**2/4 ) # ...or imaginary      

  
#@njit(fastmath=True,cache=True)
def K2inv(E,kvec,l,m,Pvec,f_kcot,IPV=0):
  k = sums.norm(kvec)
  omk = defns.omega(k)
  Pk = sums.norm(kvec-Pvec)
  E2star = E2k(E,k,Pk)
  qk2 = qst2(E,k,Pk)
  h = sums.hh(E,k,Pk)
  aux= 1.0
  if h==0:
      aux=0.0
      return 0.0      
  if l==m==0:
    kcot = f_kcot(qk2)
    out = 1/(32*pi*omk*E2star) * ( kcot + np.sqrt(np.sqrt(qk2**2))*(1-h)) - h*IPV/(32*pi*2*omk)
  else:
    return 0

  if out.imag > 1e-15:
    print('Error in K2inv: imaginary part in output')
  else:
    out = out.real
  return out


# Just compute l'=l=0 portion of K2inv
#@jit()
def K2inv_mat00_nnP(E,L,nnP,kcot,IPV=0):
  nklist = defns.list_nnk_nnP(E,L,nnP)
  K2inv00 = []
  for nkvec in nklist:
    kvec = [i*2*pi/L for i in nkvec]
    K2inv00.append(K2inv(E,np.array(kvec),0,0,np.array(nnP)*2*pi/L,kcot,IPV))
  return np.diag(K2inv00)




