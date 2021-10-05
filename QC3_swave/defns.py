import numpy as np
pi=np.pi; conj=np.conjugate; LA=np.linalg
from itertools import permutations as perms
from numba import jit,njit

####################################################################################
# This file defines several basic functions that get called multiple times
####################################################################################

@jit(fastmath=True)
def truncate(M):
    Maux = M[~np.all(M == 0, axis=1)]
    return Maux[:,~np.all(Maux == 0, axis=0)]

def truncate_idx(M):
    i_rows = ~np.all(M==0,axis=1)
    i_cols = ~np.all(M==0,axis=0)
    return (i_rows, i_cols)


@jit(nopython=True,fastmath=True)
def sqrt(x):
    return np.sqrt(x)

@jit(nopython=True,fastmath=True)
def square(x):
    return x**2


# om_k
@njit(fastmath=True,cache=True)
def omega(k):
	return sqrt( k**2 + 1 )

# E2k*
@njit(fastmath=True,cache=True)
def E2k(E,k):
	return sqrt( 1 + E**2 - 2 * E * omega(k) ) # should always be >=0

# E2k*(Pvec)
#@njit(fastmath=True,cache=True)
def E2k2_P(E,Pvec,kvec):
  k = LA.norm(kvec)
  return (E-omega(k))**2 - LA.norm(np.array(Pvec)-np.array(kvec))**2  # should always be >=0

# qk*
@jit(fastmath=True,cache=True)
def qst(E,k):
  Estar = E2k(E,k)
  if Estar >= 2:
    return sqrt( Estar**2/4 - 1 ) # can be real...
  else:
    return 1j*sqrt( 1-Estar**2/4 ) # ...or imaginary


@njit(fastmath=True,cache=True)
def qst2(E,k):
  Estar = E2k(E,k)
  if Estar >= 2:
    return  Estar**2/4 - 1  # can be real...
  else:
    return -1*( 1-Estar**2/4 ) # ...or imaginary


# gam_k
@njit(fastmath=True,cache=True)
def gamma(E, k):
    return (E - sqrt(1. + k**2))/E2k(E,k)


# List first n free energies for given L
def E_free_list(L,count,*args):
  if len(args)==0:
    nmax = 5
  else:
    nmax = args[0]

  nvec_list = []
  for n1 in range(-nmax,nmax+1):
    for n2 in range(-nmax,nmax+1):
      for n3 in range(-nmax,nmax+1):
        nvec_list.append([n1,n2,n3])

  out = {}
  for i1 in range(len(nvec_list)):
    for i2 in range(i1,len(nvec_list)):
      m1_2 = sum([x**2 for x in nvec_list[i1]])
      m2_2 = sum([x**2 for x in nvec_list[i2]])
      m12_2 = sum([x**2 for x in [-nvec_list[i1][i]-nvec_list[i2][i] for i in range(3)]])

      E1 = sqrt((2*pi/L)**2*m1_2+1)
      E2 = sqrt((2*pi/L)**2*m2_2+1)
      E12 = sqrt((2*pi/L)**2*m12_2+1)

      out[tuple(sorted([m1_2,m2_2,m12_2]))] = E1+E2+E12

  tmp = sorted(out.items(), key=lambda x: x[1])[0:count]
  out={}
  for i in tmp:
    out[i[0]]=i[1]
  return out


# List first n energies where q*=0 for given L (these give "false poles" unless the explicit q* factors in F3 are removed)
def false_poles(L,n2max):
  out = []
  if n2max<=6:
    for n2 in range(n2max+1):
      out.append(sqrt((2*pi/L)**2*n2 + 1) + sqrt((2*pi/L)**2*n2 + 4))
  else:
    nmax = floor(sqrt(n2max))
    for a in range(nmax+1):
      for b in range(a,nmax+1):
        for c in range(b,nmax+1):
          n2 = a**2+b**2+c**2
          if n2<=n2max:
            out.append(sqrt((2*pi/L)**2*n2 + 1) + sqrt((2*pi/L)**2*n2 + 4))
  return out


# List first n 2-pt. free energies for given L (not sure these are relevant)
def E_2pt_free_list(L,count,nmax=5):
  nvec_list = []
  for n1 in range(-nmax,nmax+1):
    for n2 in range(-nmax,nmax+1):
      for n3 in range(-nmax,nmax+1):
        nvec_list.append([n1,n2,n3])

  out = {}
  for i1 in range(len(nvec_list)):
      m1_2 = sum([x**2 for x in nvec_list[i1]])

      E1 = sqrt((2*pi/L)**2*m1_2+1)

      out[m1_2] = 2*E1

  tmp = sorted(out.items(), key=lambda x: x[1])[0:count]
  out={}
  for i in tmp:
    out[i[0]]=i[1]
  return out


##########################################################
# Q matrix:  diagonal in k,l,m with Q(k,l,m) = (qk*)^l
def Qmat(E,L):
  out=[]
  for shell in shell_list(E,L):
    q = qst(E,LA.norm(shell))
    out = out + ([1.]+5*[(q**2).real]) * len(shell_nnk_list(shell))
  return np.diag(out)


##########################################################
# Convert block-matrix index to (l,m)
@jit(nopython=True,fastmath=True) #FRL, it speeds up a bit. I changed the error condition to make it compatible with numba.
def lm_idx(i):
  i = i%6
  if i==0:
    [l,m] = [0,0]
  else:
    [l,m] = [2,i-3]
  return [l,m]

# Replace small real numbers in array with zero
def chop(arr):
  arr = np.array(arr)
  arr[abs(arr)<1e-13]=0
  return arr


####################################################################################
# Create nnk_list, etc.
####################################################################################
# Find maximum allowed norm(k)
# TB: Copied from Fernando's defs.pyx
@jit(nopython=True,fastmath=True,cache=True)
def kmax(E):
  alpH = -1.
  aux1 = (1. + alpH)/4.
  aux2 = (3. - alpH)/4.
  xmin = 0.02
  return sqrt((((E**2+1)/4-aux2*xmin-aux1)*(2/E))**2-1)


@jit(nopython=True,fastmath=True,cache=True)
def kmax_nnP(E):
  x1 = 0.02
  return sqrt( (E -sqrt(4.*x1))**2  -1. )



# Find energy where a certain shell turns on for given L
# Note: This is only exact if xmin=0 in kmax(E)
def Emin(shell,L,alpH=-1,xmin=0.01):
  c = (3-alpH)*xmin + (1+alpH)
  k = LA.norm([x*2*pi/L for x in shell])
  return omega(k) + sqrt(k**2+c)


# Create list of shells/orbits
# Permutation conventions: 000, 00a, aa0, aaa, ab0, aab, abc
@jit(fastmath=True,cache=True)
def shell_list(E,L):
  #nmaxreal = kmax(E)*L/(2*pi)
  nmaxreal = kmax_nnP(E)*L/(2*pi)
  nmax = int(np.floor(nmaxreal))
  shells = []
  for n1 in range(nmax+1):
    for n2 in range(n1,nmax+1):
      for n3 in range(n2,nmax+1):
        if square(n1)+square(n2)+square(n3) <= square(nmaxreal):
          # need to permute for aa0, ab0, aab
          if (n1==0<n2) or (n1>0 and n2==n3):
            shells.append((n2,n3,n1))
          else:
            shells.append((n1,n2,n3))
  # Sort by magnitude
#  shells = sorted(shells, key=lambda k: LA.norm(k))
  shells = sorted(shells, key= LA.norm)
#  print(shells)
  return shells



#@jit(fastmath=True,cache=True)
def shell_list_nnP(E,L,nnP):
  nmaxreal = kmax_nnP(E)*L/(2*pi) #+ LA.norm(nnP) #+ 1
  nmax = int(np.floor(nmaxreal))
  shells = []
  for n1 in range(nmax+1):
    for n2 in range(n1,nmax+1):
      for n3 in range(n2,nmax+1):
        if square(n1)+square(n2)+square(n3) <= square(nmaxreal):
          # need to permute for aa0, ab0, aab
          if (n1==0<n2) or (n1>0 and n2==n3):
            shells.append((n2,n3,n1))
          else:
            shells.append((n1,n2,n3))
  # Sort by magnitude
#  shells = sorted(shells, key=lambda k: LA.norm(k))
  shells = sorted(shells, key= LA.norm)

  out=[]
  for shell in shells:
    if len(shell_nnk_nnP_list(E,L,nnP,shell))>0:  # TB: this is really gross & inefficient, but gets rid of empty shells
      out.append(shell)

#  print(shells)
  #return shells
  return out




# Find where new shells open up & # of eigs increases
def shell_breaks(E,L):
  out = []
  for shell in shell_list(E,L):
    out.append(Emin(shell,L))
  #out.append(E)
  return out


# Create list of all permutations & all perms w/ 1 negation
def perms_list(nnk):
  a=nnk[0]; b=nnk[1]; c=nnk[2]
  p_list = list(perms((a,b,c)))
  p_list += list(perms((a,b,-c)))
  p_list += list(perms((a,-b,c)))
  p_list += list(perms((-a,b,c)))

  p_list = [ p for p in p_list ]
  return p_list


# Create list of all nnk in a given shell
#@jit()
def shell_nnk_list(shell):
  # 000
  if list(shell)==[0,0,0]:
    return [shell]

  # 00a
  elif shell[0]==shell[1]==0<shell[2]:
    a = shell[2]
    return [(0,0,a),(0,a,0),(a,0,0),(0,0,-a),(0,-a,0),(-a,0,0)]

  # aa0
  elif shell[0]==shell[1]>0==shell[2]:
    a = shell[0]
    return [(a,a,0),(a,0,a),(0,a,a),(a,-a,0),(a,0,-a),(0,a,-a),    (-a,-a,0),(-a,0,-a),(0,-a,-a),(-a,a,0),(-a,0,a),(0,-a,a)]

  # aaa
  elif shell[0]==shell[1]==shell[2]>0:
    a = shell[0]
    return [(a,a,a),(a,a,-a),(a,-a,a),(-a,a,a), (-a,-a,-a),(-a,-a,a),(-a,a,-a),(a,-a,-a)]

  # ab0
  elif 0==shell[2]<shell[0]<shell[1]:
    a = shell[0]; b = shell[1]
    return [
      (a,b,0),(b,a,0),(a,0,b),(b,0,a),(0,a,b),(0,b,a),
      (a,-b,0),(-b,a,0),(a,0,-b),(-b,0,a),(0,a,-b),(0,-b,a),
      (-a,-b,0),(-b,-a,0),(-a,0,-b),(-b,0,-a),(0,-a,-b),(0,-b,-a),
      (-a,b,0),(b,-a,0),(-a,0,b),(b,0,-a),(0,-a,b),(0,b,-a)
    ]

  # aab
  elif 0<shell[0]==shell[1]!=shell[2]>0:
    a = shell[0]; b = shell[2]
    return [
      (a,a,b),(a,b,a),(b,a,a),
      (a,a,-b),(a,-b,a),(-b,a,a),
      (a,-a,b),(a,b,-a),(b,a,-a),
      (-a,a,b),(-a,b,a),(b,-a,a),

      (-a,-a,-b),(-a,-b,-a),(-b,-a,-a),
      (-a,-a,b),(-a,b,-a),(b,-a,-a),
      (-a,a,-b),(-a,-b,a),(-b,-a,a),
      (a,-a,-b),(a,-b,-a),(-b,a,-a)
    ]

  # abc
  elif 0<shell[0]<shell[1]<shell[2]: #FRL There was a typo here. Half of the shell was missing.
    auxshell1 = perms_list(shell)
    auxshell2 = 1*auxshell1
    for i in range(len(auxshell1)):
        auxshell2[i] = tuple([x*-1 for x in auxshell1[i]])
#    print(auxshell1,auxshell2)
    return auxshell1+auxshell2

  else:
    print('Error in shell_nnk_list: Invalid shell input')

def shell_nnk_nnP_list(E,L,nnP,shell,xmin=0.02):
  Pvec = np.array(nnP)*2*pi/L
  nnk_list=[]
  for nnk in shell_nnk_list(shell):   # TB: just do this instead of truncate()
    kvec = np.array(nnk)*2*pi/L
    k = LA.norm(kvec)
    if E2k2_P(E,Pvec,kvec) > 4*xmin:  # TB: xmin needs to be same as in jj()
      #print(nnk,LA.norm(nnk))
      nnk_list.append(nnk)
  return nnk_list

# New nnk_list broken into shells
def list_nnk(E,L):
  nnk_list = []
  for shell in shell_list(E,L):
    nnk_list += shell_nnk_list(shell)
  return nnk_list

def list_nnk_nnP(E,L,nnP):
  nnk_list = []
  for shell in shell_list_nnP(E,L,nnP):
    #nnk_list += shell_nnk_list(shell)
    nnk_list += shell_nnk_nnP_list(E,L,nnP,shell) # TB: use this instead of truncate()
  return nnk_list


# Make list of all nkvecs that contribute; i.e., H(kvec)!=0 for each kvec = nkvec*2*pi/L
# TB: Copied from Fernando's F2_alt.py
def list_nnk_old(E,L):
  twopibyL = 2*pi/L
  nkmaxreal = kmax(E)/twopibyL;

  # maximum magnitude of nk that still contributes; i.e., H(k)!=0
  nkmax = int(np.floor(nkmaxreal))

  nklist = []
  for n1 in range(-nkmax,nkmax+1):
    for n2 in range(-nkmax,nkmax+1):
      for n3 in range(-nkmax,nkmax+1):
        if np.sqrt(n1**2+n2**2+n3**2)<nkmaxreal:
          nklist.append([n1,n2,n3])
  return nklist


####################################################################################
# Spherical harmonics
####################################################################################

# Complex spherical harmonics
@jit(nopython=True,fastmath=True) #FRL
def y2complex(kvec,m): # y2 = |kvec|**2 * sqrt(4pi) * Y2
  if m==2:
    return sqrt(15/8)*(kvec[0]+1j*kvec[1])**2
  elif m==-2:
    return sqrt(15/8)*(kvec[0]-1j*kvec[1])**2
  elif m==1:
    return -sqrt(15/2)*(kvec[0]+1j*kvec[1])*kvec[2]
  elif m==-1:
    return sqrt(15/2)*(kvec[0]-1j*kvec[1])*kvec[2]
  elif m==0:
    return sqrt(5/4)*(2*kvec[2]**2-kvec[0]**2-kvec[1]**2)
  else:
    print('Error: invalid m input in y2')

# Real spherical harmonics
@jit(nopython=True,fastmath=True) #FRL
def y2real(kvec,m): # y2 = sqrt(4pi) * |kvec|**2 * Y2
  if m==-2:
    return sqrt(15)*kvec[0]*kvec[1]
  elif m==-1:
    return sqrt(15)*kvec[1]*kvec[2]
  elif m==0:
    return sqrt(5/4)*(2*square(kvec[2])-square(kvec[0])-square(kvec[1]))
  elif m==1:
    return sqrt(15)*kvec[0]*kvec[2]
  elif m==2:
    return sqrt(15/4)*(square(kvec[0])-square(kvec[1]))
  else:
    print('Error: invalid m input in y2real')

# Spherical harmonics w/ flag for real (default) vs. complex
#@jit(nopython=True,fastmath=True)
def y2(kvec,m,Ytype='r'):
  if Ytype=='real' or Ytype=='r':
    return y2real(kvec,m)
  else:
    return y2complex(kvec,m)

# Generalize to include l=0 case
#@jit(nopython=True,fastmath=True) #FRL
def ylm(kvec,l,m,Ytype='r'):
  if l==m==0:
    return 1
  elif l==2:
    return y2(kvec,m,Ytype)
  else:
    print('Error: ylm can only take l=0 or l=2')




####################################################################################
# Graveyard (old functions)
####################################################################################

# Convert (l'm',lm) matrix at fixed (p,k) from complex to real basis (not used anymore)
def cpx2real(cpx_mat):
  U = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1j/sqrt(2), 0, 0, 0, -1j/sqrt(2)],
    [0, 0, 1j/sqrt(2), 0, 1j/sqrt(2), 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 1/sqrt(2), 0, -1/sqrt(2), 0],
    [0, 1/sqrt(2), 0, 0, 0, 1/sqrt(2)]
  ])
  Ui = U.conj().T

  # block size
  N = int(len(cpx_mat)/6)

  real_mat = np.zeros((len(cpx_mat),len(cpx_mat)),dtype=complex)
  for ip in range(N):
    for ik in range(N):
      mat_pk = cpx_mat[ip::N,ik::N]
      real_mat[ip::N,ik::N] = U @ mat_pk @ Ui

  if ((abs(real_mat.imag))>1e-15).any():
    print('Error: nonzero imaginary part in output of cpx2real')
  else:
    real_mat = real_mat.real

  return real_mat


####################################################################################
# Combine (l'm',lm) blocks into full matrix (old structure)
def full_matrix(M00,M20,M02,M22):
  return np.vstack((np.hstack((M00,M02)),np.hstack((M20,M22))))



@jit(nopython=True, fastmath=True) 
def norm2(nnk):
    nk=0.
    for i in nnk:
        nk += i**2
    return nk


def multiplicity_nnk(nnk_list, nnP):
  nnk_list_new=[nnk_list[0]]
  multiplicity=[0]
  norm_sub_tmp=norm2(np.array(nnk_list[0])-np.array(nnP))
  norm_tmp=norm2(np.array(nnk_list_new[0]))
  i=0
  for nnk in nnk_list:
    found=False
    norm_sub=norm2(np.array(nnk)-np.array(nnP))
    norm1=norm2(np.array(nnk))
    if norm_sub==norm_sub_tmp  and  norm1 == norm_tmp :    
      found=True
    
    if  found :
      multiplicity[i]+=1
    else:
      nnk_list_new+=[nnk]
      multiplicity+=[1]
      norm_sub_tmp=norm_sub
      norm_tmp=norm1 
      i+=1
  return nnk_list_new, multiplicity



def short_nnk_list(nnk_list, nnP):
  N=len(nnk_list)
  if N==0
    return [], [[]]

  nnk_list_new=[nnk_list[0]]
  multiplicity=[[]]
  norm_sub_tmp=[norm2(np.array(nnk_list[0])-np.array(nnP))]
  norm_nnk_tmp=[norm2(np.array(nnk_list_new[0]))]
  j=0
  
  for i in range(N):
    found=False
    norm_sub=norm2(np.array(nnk_list[i])-np.array(nnP))
    norm_nnk=norm2(np.array(nnk_list[i]))
    N_tmp=len(norm_sub_tmp)
    for id_n in range(N_tmp):
        if norm_sub==norm_sub_tmp[id_n]  and  norm_nnk == norm_nnk_tmp[id_n] :    
          found=True
          break
      
    
    if  found :
      multiplicity[id_n]+=[i]
    else:
      nnk_list_new+=[nnk_list[i]]
      multiplicity+=[[i]]
      norm_sub_tmp+=[norm_sub]
      norm_nnk_tmp+=[norm_nnk]
      j+=1
  return nnk_list_new, multiplicity