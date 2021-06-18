import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
from scipy.linalg import block_diag

import defns, group_theory_defns as GT
import sys

###########################################################
# Here we implement various projections, including the isotropic approx, A1+, and all other irreps
###########################################################
# Pull out l'=l=0 part of full matrix M
def l0_proj(M):
  N = int(len(M)/6)
  M00 = np.zeros((N,N))
  for i in range(N):
    I = 6*i
    for j in range(N):
      J = 6*j
      M00[i][j] = M[I][J]
  return M00

# Pull out l'=l=2 part of full matrix M
def l2_proj(M):
  N = int(len(M)/6)
  M22 = np.zeros((5*N,5*N))
  for i in range(N):
    i_min = 5*i;   i_max = 5*(i+1)
    I_min = 6*i+1; I_max = 6*(i+1)
    for j in range(N):
      j_min = 5*j;   j_max = 5*(j+1)
      J_min = 6*j+1; J_max = 6*(j+1)

      M22[i_min:i_max,j_min:j_max] = M[I_min:I_max,J_min:J_max]
  return M22

###########################################################
# Isotropic approx

# Matrix projecting l=0 matrix onto A1 for iso approx
# Note: normalization convention seems irrelevant
def p_iso00(E,L):
  # No normalization
#  p_iso = np.ones((len(defns.list_nnk(E,L)),1))

  # Steve's convention: each column corresponds to a shell with 1/sqrt(N) normalization (so each vector has length=1)
  shells = defns.shell_list(E,L)
  p_iso = np.zeros((len(defns.list_nnk(E,L)),len(shells)))
  i_k = 0
  for i_shell in range(len(shells)):
    N = len(defns.shell_nnk_list(shells[i_shell]))
    p_iso[i_k:i_k+N,i_shell] = 1/sqrt(N)
    i_k += N
#  print(p_iso)
#  print(np.sum(p_iso,axis=1))
#  return np.sum(p_iso,axis=1)
  return p_iso

# Project l'=l=0 matrix into isotropic approx (A1 irrep & l=0)
def iso_proj00(M00,E,L):
  p_iso = p_iso00(E,L)
  return defns.chop( p_iso.T @ M00 @ p_iso )



####################################################################################
# Projection onto ANY irrep of LG(nnP) for given nnP
####################################################################################
# Full projection matrices (P^I in notes)

# Single shell & single l for given nnP
def P_irrep_o_l(E,L,nnP,I,shell,l,lblock=False):
  nnk_list = defns.shell_nnk_nnP_list(E,L,nnP,shell)
  Lk = GT.little_group(shell)
  d_I = GT.irrep_dim(I)

  nnP=list(nnP)
  if nnP==[0,0,0]:
    P_shell = []
    for k2 in nnk_list:
      R20 = GT.Rvec(k2,shell)
      P_k2 = []
      for k1 in nnk_list:
        R01 = GT.Rvec(shell,k1)

        P_block = np.zeros((6,6))
        for R in Lk:
          RRR = GT.R_prod(R20,R,R01)
          if l==0:
            P_block[0,0] += GT.chi(RRR,I,nnP)
          elif l==2:
            P_block[1:,1:] += GT.chi(RRR,I,nnP) * GT.Dmat(RRR)[1:,1:]
            P_block = defns.chop(P_block)

        P_k2.append(P_block)
      P_shell.append(P_k2)

    out = d_I/48 * np.block(P_shell)
    if lblock==True:
      if l==0:
        return l0_proj(out)
      elif l==2:
        return l2_proj(out)
    else:
      return out

  else:
    P_shell = []
    LG = GT.little_group(nnP)
    for k2 in nnk_list:
      #R20 = GT.Rvec(k2,shell)
      #print(k2)
      P_k2 = []
      for k1 in nnk_list:
        #R01 = GT.Rvec(shell,k1)

        P_block = np.zeros((6,6))
        for R in LG:
          Rk1 = [np.sign(R[i])*k1[abs(R[i])-1] for i in range(3)]
          #print(Rk1)
          if Rk1 == list(k2):
            #print(Rk1)
            if l==0:
              #print(R,I,nnP)
              P_block[0,0] += GT.chi(R,I,nnP)
            elif l==2:
              P_block[1:,1:] += GT.chi(R,I,nnP) * GT.Dmat(R)[1:,1:]
              P_block = defns.chop(P_block)

        #print(P_block)
        P_k2.append(P_block)
      #print(P_k2)
      P_shell.append(P_k2)
    #print(shell,P_shell)

    out = d_I/len(LG) * np.block(P_shell)
    #print(out.shape)
    if lblock==True:
      if l==0:
        return l0_proj(out)
      elif l==2:
        return l2_proj(out)
    else:
      return out


# Projection for a given shell (contains l=0 and l=2)
def P_irrep_o(E,L,nnP,I,shell):
  return P_irrep_o_l(E,L,nnP,I,shell,0) + P_irrep_o_l(E,L,nnP,I,shell,2)


# Full projection matrix (includes all shells & l)
def P_irrep_full(E,L,nnP,I):
  P_block_list = []
  for shell in defns.shell_list_nnP(E,L,nnP):
    P_block_list.append( P_irrep_o(E,L,nnP,I,shell) )
  return block_diag(*P_block_list)

############################################
# l'=l=0 part of projection matrix
def P_irrep_00(E,L,nnP,I):
  return l0_proj(P_irrep_full(E,L,nnP,I))

# l'=l=2 part of projection matrix
def P_irrep_22(E,L,nnP,I):
  return l2_proj(P_irrep_full(E,L,nnP,I))


###################################################################
# Subspace projection

# Irrep subspace projection matrix onto specific shell & l (acts on full matrix)
def P_irrep_subspace_o_l(E,L,nnP,I,shell,l,lblock=False):

  P_block_list = []
  for nk in defns.shell_list_nnP(E,L,nnP):
    if list(nk) == list(shell):
      P_block_list.append( P_irrep_o_l(E,L,nnP,I,shell,l,lblock) )
    else:
      P_block_list.append( np.zeros(P_irrep_o_l(E,L,nnP,I,nk,l,lblock).shape ))

  P_I = block_diag(*P_block_list)

  elist, vlist = LA.eig(P_I)

  # eigenvalues should all be 0 or 1; only want 1's
  ivec = [i for i,e in enumerate(elist) if abs(e-1)<1e-13]
  if len(ivec) != int(round(np.trace(P_I))):
    print('Error in P_irrep_subspace: wrong subspace dimension')
  Psub = defns.chop(vlist[:,ivec].real)
  if ivec==[]:
    return Psub
  else:
    #print(defns.chop(LA.qr(Psub)[0]))
    return defns.chop(LA.qr(Psub)[0])


# Irrep projection matrix onto specific shell (acts on full matrix, contains l=0 and l=2)
def P_irrep_subspace_o(E,L,nnP,I,shell):
  return np.concatenate((P_irrep_subspace_o_l(E,L,nnP,I,shell,0),P_irrep_subspace_o_l(E,L,nnP,I,shell,2)),axis=1)


# Irrep subspace projection matrix (includes all shells & l)
def P_irrep_subspace(E,L,nnP,I):
  # P_I = P_irrep_full(E,L,I)
  # elist, vlist = LA.eig(P_I)
  # # eigenvalues should all be 0 or 1; only want 1's
  # ivec = [i for i,e in enumerate(elist) if abs(e-1)<1e-13]
  # if len(ivec) != int(round(np.trace(P_I))):
  #   print('Error in P_irrep_subspace: wrong subspace dimension')
  # Psub = defns.chop(vlist[:,ivec].real)
  # if Psub.shape[1]==0:
  #   return Psub
  # else:
  #   return defns.chop(LA.qr(Psub)[0]) # orthonormalize

  N = len(defns.list_nnk_nnP(E,L,nnP))
  Psub = np.zeros((6*N,0))
  for shell in defns.shell_list_nnP(E,L,nnP):
    Psub = np.column_stack((Psub,P_irrep_subspace_o(E,L,nnP,I,shell)))
  return Psub



# Project matrix onto irrep subspace (all shells & l)
def irrep_proj(M,E,L,nnP,I):
  P_I = P_irrep_subspace(E,L,nnP,I)
  return defns.chop( P_I.T @ M @ P_I )



########################################
# l'=l=0 irrep subspace projection matrix (acts on F3_00)
def P_irrep_subspace_00(E,L,nnP,I):
  N = len(defns.list_nnk_nnP(E,L,nnP))
  Psub = np.zeros((N,0))
  for shell in defns.shell_list_nnP(E,L,nnP):
    Pnew = P_irrep_subspace_o_l(E,L,nnP,I,shell,0,lblock=True)
    Psub = np.column_stack((Psub,Pnew))
    #Psub = np.column_stack((Psub,P_irrep_subspace_o_l(E,L,nnP,I,shell,0,lblock=True)))
  return Psub

# l'=l=2 irrep subspace projection matrix (acts on F3_00)
def P_irrep_subspace_22(E,L,nnP,I):
  N = len(defns.list_nnk_nnP(E,L,nnP))
  Psub = np.zeros((5*N,0))
  for shell in defns.shell_list_nnP(E,L,nnP):
    Psub = np.column_stack((Psub,P_irrep_subspace_o_l(E,L,nnP,I,shell,2,lblock=True)))
  return Psub


# Project l'=l=0 matrix onto irrep
def irrep_proj_00(M00,E,L,nnP,I):
  Psub = P_irrep_subspace_00(E,L,nnP,I)
  return defns.chop(Psub.T @ M00 @ Psub)

# Project l'=l=2 matrix onto irrep
def irrep_proj_22(M22,E,L,nnP,I):
  Psub = P_irrep_subspace_22(E,L,nnP,I)
  return defns.chop(Psub.T @ M22 @ Psub)


####################################################################################
# Eigenvector decomposition by o,l for given irrep
def evec_decomp(v,E,L,nnP,I):
  c0_list =[]
  c2_list = []
  shells = defns.shell_list_nnP(E,L,nnP)
  for shell in shells:
    P0 = P_irrep_subspace_o_l(E,L,nnP,I,shell,0)
    P2 = P_irrep_subspace_o_l(E,L,nnP,I,shell,2)

    c0 = sum([np.dot(v,P0[:,i])**2 for i in range(P0.shape[1])])/LA.norm(v)**2
    c2 = sum([np.dot(v,P2[:,i])**2 for i in range(P2.shape[1])])/LA.norm(v)**2

    c0_list.append(c0)
    c2_list.append(c2)

  s = sum(c0_list)+sum(c2_list)
  print('Eigenvector decomposition for '+str(I)+' (total fraction: '+str(round(s,6))+')')
  for i in range(len(shells)):
    if s==0:
      frac0=0.; frac2=0.
    else:
      frac0 = c0_list[i]/s
      frac2 = c2_list[i]/s
    print(shells[i],'--- l=0:',round(frac0,8),',\t l=2:',round(frac2,8))
  print()

  return s


# Irrep decomposition of large or small eigenvalues (e.g., poles of F3)
def pole_decomp(M,E,L,nnP,size='large',thresh=100):
  out = {}
  for I in GT.irrep_list(nnP):
    eigs_I = LA.eigvals(irrep_proj(M,E,L,nnP,I)).real
    if size=='large':
      eigs_I = [e for e in eigs_I if abs(e)>=thresh]
    else:
      eigs_I = [e for e in eigs_I if abs(e)<=thresh]
    out[I] = len(eigs_I)
    print(I+':\t' + str(len(eigs_I)) + '\t' + str(eigs_I)) # comment out if desired
  print()                                                  # comment out if desired
  return out


####################################################################################
# Sum over Wigner D-matrices needed for A1+ projection matrix (see notes)
def A1_little_group_sum(shell):
  Lk = GT.little_group(shell)
  Usum = np.zeros((6,6))
  for R in Lk:
    Usum += GT.Dmat(R)
  Usum = defns.chop( Usum/len(Lk) )
  return Usum



#####################################################################
# Graveyard (old code)
#####################################################################
# These first two functions aren't useful since F3 connects different shells & \ell's

# Project matrix onto irrep subspace for single shell & l
def irrep_proj_o_l(M,E,L,I,shell,l):
  P_I = P_irrep_subspace_o_l(E,L,I,shell,l)
  return defns.chop( P_I.T @ M @ P_I )

# Project matrix onto irrep subspace for single shell (contains l=0 and l=2)
def irrep_proj_o(M,E,L,I,shell):
  P_I = P_irrep_subspace_o(E,L,I,shell)
  return defns.chop( P_I.T @ M @ P_I )


#######################################################
# Old A1+ projection code
# NOTE: All of this code should work, but it's now redundant (above code can project onto any irrep)

# Full A1+ projection matrix for a given shell
# Naive method: brute force sum
def P_A1_o_naive(shell):
  Nk = len(defns.shell_nnk_list(shell))
  P_shell = np.zeros((6*Nk,6*Nk))
  for R in GT.Oh_list():
    P_shell += np.kron(GT.Smat(R,shell).T,GT.Dmat(R))
  P_shell = P_shell/48

  return P_shell


# Smarter version only summing over L_k
def P_A1_o(shell):
  shell = list(shell)
  No = len(defns.shell_nnk_list(shell))

  one_mat = np.ones((No,No))/No
  Pk = A1_little_group_sum(shell)

  U_block_list = []
  for p in defns.shell_nnk_list(shell):
    p = list(p)
    U_block_list.append(GT.Dmat(GT.Rvec(p,shell)))
  U = block_diag(*U_block_list)

  return U @ np.kron(one_mat,Pk) @ U.T


# Full A1 projection matrix (includes all shells)
def P_A1_full(E,L):
  P_block_list = []
  for shell in defns.shell_list(E,L):
    P_block_list.append( P_A1_o(shell) )
  return block_diag(*P_block_list)


# A1 subspace projection matrix (includes all shells)
def P_A1_subspace(E,L):
  P_A1 = P_A1_full(E,L)
  elist, vlist = LA.eig(P_A1)
  # eigenvalues should all be 0 or 1; only want 1's
  ivec = [i for i,e in enumerate(elist) if abs(e-1)<1e-13]
  if len(ivec) != int(round(np.trace(P_A1))):
    print('Error in P_A1_subspace: wrong subspace dimension')
  return defns.chop(vlist[:,ivec].real)


# Project matrix onto A1 irrep subspace
def A1_proj(M,E,L):
  P_A1 = P_A1_subspace(E,L)
  return defns.chop( P_A1.T @ M @ P_A1 )

############################################
# A1+, 2 shells (even older than above code)

# Create A1 projection matrix for 2 shells
def p_A1_2shells():
  pA1 = np.zeros((42,3))
  pA1[0][0] = 1
  for i in range(1,7):
    pA1[6*i][1] = 1/sqrt(6)
    pA1[6*i+3][2] = 1/sqrt(6)
  return pA1


def p_A1_old(E,L):
  Nshells = len(defns.shell_list(E,L))
  if Nshells==1:
    pA1 = np.array([1,0,0,0,0,0]).T
  elif Nshells==2:
    pA1 = p_A1_2shells()
  return pA1


# Project matrix M onto A1 irrep (old method)
def A1_proj_old(M,E,L):
  pA1 = p_A1_old(E,L)
  return defns.chop( pA1.T @ M @ pA1 )


###############################################
# Iso approx using l=0 part of A1 projection
# NOTE: THIS DOESN'T WORK! -- must do l=0 for every matrix, not just F3
def iso_proj_bad(M,E,L):
  M00 = l0_proj(M) # BAD!
  return iso_proj00(M00,E,L)
