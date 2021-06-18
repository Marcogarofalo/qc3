import numpy as np
pi = np.pi
import math, sys
import defns

from numpy.lib.scimath import sqrt


#from pathlib import Path
from numba import jit,njit
from scipy.special import sph_harm
from scipy.special import erfi
from scipy.special import erfc
from scipy.optimize import fsolve

@jit(nopython=True,fastmath=True,cache=True)
def npsqrt(x):
    return np.sqrt(x)

@jit(nopython=True,fastmath=True,cache=True)
def square(x):
    return x**2

#This is an asymptotic expansion of erfc function. Numba doesn't accept scipy.especial.erfc
@njit(fastmath=True,cache=True)
def myerfc(x):
    return exp(-square(x))/npsqrt(math.pi)/x*(1.- 1./2/square(x) + 3./square(2*square(x)))


@jit(nopython=True,fastmath=True,cache=True)
def exp(x):
    return np.exp(x)

@jit(nopython=True,fastmath=True)
def mydot(x,y):
    res = 0.
    for i in range(3):
        res+=x[i]*y[i]
    return res

##temporal
@jit(nopython=True,fastmath=True,cache=True)
def jj( x):
    xmin = 0.02
    xmax = 0.97
    if xmin < x < xmax:
        return exp(-exp(-1/(1-x))/x)
    elif x >= xmax:
        return 1.
    else:
        return 0.


@jit(nopython=True,fastmath=True,cache=True)
def E2a2(e, a,Pk):
    return ((e - npsqrt(1+a**2))**2 - Pk**2)/4  # TB: Note to self -- this is really (omega_a*)^2 = (E_{2,a}*/2)^2


@jit(nopython=True, fastmath=True) 
def norm(nnk):
    nk=0.
    for i in nnk:
        nk += i**2
    return npsqrt(nk)

@jit(nopython=True, fastmath=True)
def hh(e, k,Pk):
    alpH = -1.
    aux1 = (1. + alpH)/4.
    aux2 = (3. - alpH)/4.
    return jj( (E2a2(e,k,Pk) - aux1)/aux2  )

@jit(nopython=True, fastmath=True)
def hhq(q2):
    alpH = -1.
    aux1 = (1. + alpH)/4.
    aux2 = (3. - alpH)/4.
    E2a2_ =    (npsqrt(1 + q2))**2
    return jj( (E2a2_ - aux1)/aux2  )

##temporal
#@jit(nopython=True,fastmath=True,cache=True)
#def gam(e, k):
#    return (e - npsqrt(1. + k**2))/(2*npsqrt(E2a2(e, k)))


@jit(nopython=True,fastmath=True,cache=True)
def gam(e, k,Pk):
    return (e - npsqrt(1. + k**2))/(2*npsqrt(E2a2(e, k,Pk)))


##temporal
@jit(nopython=True,fastmath=True,cache=True)
def xx2(e, L, k,Pk):
    return ( E2a2(e, k,Pk) - 1)*square(L/(2*math.pi));


@jit(nopython=True,fastmath=True,cache=True)
def summand(e, L, nna, nnk, nPk, gamma, x2,l1,m1,l2,m2,alpha,nnP):

#    nnA = np.array(nna)
#    nnK = np.array(nnk) - np.array(nnP)

    if(nPk==0):
        rr=nna
    else:
        nnK = nnk - nnP
        factor=(1/(2*gamma)+(1/gamma -1)*mydot(nna,nnK)/square(nPk))
        rr = np.add(nna,factor*nnK)


    #    nnb = -1*(np.add(nnA,nnK))

#    if(nk==0):
#        rr=nnA
#    else:

    rr2 = mydot(rr, rr)
#    twopibyL = 2*math.pi/L
#    a = norm(nnA)*twopibyL
#    b = norm(nnb)*twopibyL

    Ylmlm=1
    if l1==2:
      #Ylmlm = defns.y2(rr,m1,Ytype)
      Ylmlm = defns.y2real(rr,m1)
    if l2==2:
      #Ylmlm = Ylmlm * defns.y2(rr,m2,Ytype)
      Ylmlm = Ylmlm * defns.y2real(rr,m2)

    exponential = exp(alpha*(x2-rr2))

    out = Ylmlm*exponential/(x2 - rr2)
    # if (Ytype=='r' or Ytype=='real') and abs(out.imag)>1e-15:
    #   sys.exit('Error in summand: imaginary part in real basis')
    return out.real


# Find maximum n needed in sum_nnk
@njit(fastmath=True) #This is compatible with numba
def getnmax2(cutoff,alpha,x2,gamma):
    n0=7
    mathpi=math.pi
    res = 2*mathpi*npsqrt(math.pi/alpha) * exp(alpha*x2)*myerfc(npsqrt(alpha)*n0)
    while(res>cutoff):
        n0+=1
        res = 2*mathpi*npsqrt(math.pi/alpha) * exp(alpha*x2)*myerfc(npsqrt(alpha)*n0)

    return int(n0*gamma+3)


# Compute sum needed for Ftilde
@njit(fastmath=True,cache=True)
def sum_nnk(e, L, nnk,l1,m1,l2,m2,alpha, nnP,smart_cutoff=0):
    nk = norm(nnk)
    nPk = norm(nnk -nnP)
    alpH = -1.
    aux1 = (1. + alpH)/4.
    

    if(E2a2(e, nk*2*math.pi/L,nPk*2*math.pi/L)<=aux1):
        return 0.
    else:
        twopibyL = 2.*math.pi/L
        k = nk*twopibyL
        Pk = nPk*2*math.pi/L
        #nn0 = np.sqrt((e**2 - alpH)**2/(4*e**2) - 1)/twopibyL;
        gamma = gam(e, k,Pk)
        x2 = xx2(e, L, k, Pk)
 #       nmax = math.floor(nn0);
        #hhk = hh(e, k)  # TB: put hhk in C

        cutoff=1e-9
        hhk = hh(e,k,Pk)
        if hhk==0:
          return 0
        if smart_cutoff==1:
          cutoff = cutoff/hhk  # TB: This should fix the run-time issue at shell thresholds (large gamma, but tiny hhk)

        nmax = getnmax2(cutoff,alpha,x2,gamma)
#aqui
        nparray=np.array
        ressum=0.
        for n1 in range(-nmax,nmax+1):
            for n2 in range(-nmax,nmax+1):
                for n3 in range(-nmax,nmax+1):
                    if(norm([n1,n2,n3])<nmax): #FRL Sphere instead of cube.
                        ressum += summand(e, L, 1.0*nparray([n1, n2, n3]), nnk, nPk, gamma, x2,l1,m1,l2,m2,alpha,nnP) #TB
                    #ressum += hhk*summand(e, L, [n1, n2, n3], nnk, gamma, x2,l1,m1,l2,m2,alpha)

        # return (x2*twopibyL**2)**(-(l1+l2)/2)*ressum # FRL
        #return x2**(-(l1+l2)/2)*ressum # TB
        return (2*pi/L)**(l1+l2) * ressum # TB, no q




#@njit(fastmath=True,cache=True)
#@autojit
#@jit(fastmath=True, cache=True)
def int_nnk(e,L,nnk,l1,m1,l2,m2,alpha,nnP):

    if(l1!=l2 or m1!=m2):
        return 0.

    elif(l1==l2==0):
        nk = norm(nnk)
        twopibyL = 2.*math.pi/L
        Pk = norm(np.array(nnk) - np.array(nnP))*twopibyL
        k = nk*twopibyL
        gamma = gam(e, k, Pk)
        x2 = xx2(e, L, k,Pk)
        x_term = x2**(-(l1+l2)/2) # TB
        factor1 = -npsqrt(math.pi/alpha)*0.5*exp(alpha*(x2))
        factor2 = 0.5*math.pi*sqrt(x2)*erfi(sqrt(alpha*x2))

        out = 4*math.pi*gamma*(factor1 + factor2) #TB, no q

#    elif(l1==l2==2):
#        nk = norm(nnk)
#        twopibyL = 2.*math.pi/L
#        k = nk*twopibyL
#        gamma = gam(e, k)
#        x2 = xx2(e, L, k)
#        x_term = x2**(-(l1+l2)/2) # TB
#
#        factor1 = -npsqrt(math.pi/alpha**5)*(3+2*alpha*x2+4*alpha**2*x2**2)*exp(alpha*(x2))/8
#        factor2 = 0.5*math.pi*sqrt(x2**5)*erfi(sqrt(alpha*x2))
#        out = (2*pi/L)**4 * 4*math.pi*gamma*(factor1 + factor2) #TB, no q

    else:
      return 0.

#    if abs(out.imag)>1e-15:
#        print('Error in int_nnk: imaginary part in output')
#    else:
#      out = out.real
    return out.real


# Calculate F (this is really Ftilde=F/(2*omega))
#@jit(fastmath=True, cache=True)
def F2KSS(e,L,nnk,l1,m1,l2,m2,alpha,nnP):
    nk = norm(nnk)
    k = nk*2*math.pi/L
    nPk = norm(np.array(nnk)-np.array(nnP))
    Pk = nPk*2*math.pi/L
    hhk = hh(e, k,Pk)

    if hhk==0:
        return 0
    else:
        omk = npsqrt(1. + k**2)
        SUM = sum_nnk(e, L, np.array(nnk),l1,m1,l2,m2,alpha,nnP)
        INT = int_nnk(e,L,nnk,l1,m1,l2,m2,alpha,nnP)
        C = hhk/(32*omk*math.pi**2*L*(e - omk)) #TB
        return (SUM-INT)*C
