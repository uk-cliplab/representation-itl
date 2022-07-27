"""
Informativeness.py
Informativeness measurements for correlation matrices
"""

import numpy as np
import torch
import repitl.divergences as div
import repitl.kernel_utils as ku
import repitl.matrix_itl_approx as approx
import repitl.matrix_itl as itl


def informativeness(K,alpha,variant = 'renyi', average = False, n_eig = 1):
    """ Computes alpha order informativeness measurements for 
    correlation matrices. In general informativeness can be defined as 
    some sort of "distance" between a correlation matrix and its "closest" 
    non-informative matrix. 
   
    Args:
      K: (N x N) Gram matrix.
      alpha: order of the entropy (norm).
      variant: type of informativeness. Options: Renyi, ratio, Tsallis, distance
      average: Boolean (default False). Chooses weather or not the average of 
               correlations is used to find the non-informative matrix
      n_eig: Defines the number of eigen-values to preserve while finding the 
             closest non-informative matrix,
    
    Returns:
      I: Informativeness. 
    """
    
    Ev, _ = torch.linalg.eigh(K)
    Ev, _ = Ev.sort(descending = True)
    mk = torch.lt(Ev, 0.0)
    Ev[mk] = 0.0
    n = Ev.shape[0]
    Ev = Ev/torch.sum(Ev)
    I = spectrumInformativeness(Ev,alpha = alpha, variant = variant, average = average, K = K, n_eig = n_eig)
    return I

def spectrumInformativeness(Ev, alpha,variant = 'renyi', average = False, K = None, n_eig = 1): 
    """ Computes alpha order spectrum informativeness for 
    a set of normalized eigen-values sorted descendingly (sum of them must be 1).
    Args:
      Ev: Set of eigen-values.
      alpha: order of the entropy (norm).
      variant: type of informativeness. Options: Renyi, ratio, Tsallis, distance
      average: Boolean (default False). Chooses weather or not the average of 
               correlations is used to find the non-informative matrix
      K: (N x N) Gram-matrix, neccesary in case of using average as criteria to
         build the non-informative matrix
      n_eig: Defines the number of eigen-values to preserve while finding the 
             closest non-informative matrix,
    
    Returns:
      I: Informativeness. 
    """    
    Evni = nonInformativeSpectrum(Ev, average = average, K = K, n_eig = n_eig)
    if variant == 'renyi':
        H   = (1/(1-alpha))*torch.log(alpha_norm(Ev,alpha))
        Hni = (1/(1-alpha))*torch.log(alpha_norm(Evni,alpha))
        I = Hni - H
    elif variant == 'ratio':
        if alpha > 1:
            I = alpha_norm(Ev,alpha)/alpha_norm(Evni,alpha)
        else:
            I = alpha_norm(Evni,alpha)/alpha_norm(Ev,alpha)
    elif variant == 'tsallis':
        q = alpha
        H   = (1/(q-1))*(1-alpha_norm(Ev,alpha))
        Hni = (1/(q-1))*(1-alpha_norm(Evni,alpha))
        I = Hni - H   
    elif variant == 'distance':
        I = alpha_norm(Ev - Evni,alpha) 
    else:
        raise ValueError(" Got a false variant value")
    return I

def informativenessRFF(Z,alpha,variant = 'renyi', n_eig = 1):
    
    """ Computes alpha order informativeness measurements for 
    a set of points mapped to the RKHS. 
   
    Args:
      K: (N x N) Gram matrix.
      alpha: order of the entropy (norm).
      variant: type of informativeness. Options: Renyi, ratio, Tsallis, distance
      n_eig: Defines the number of eigen-values to preserve while finding the 
             closest non-informative matrix,
    
    Returns:
      I: Informativeness. 
    """
   

    S = torch.linalg.svdvals(Z)
    Ev = (S**2)/Z.shape[0]
    Ev, _ = Ev.sort(descending = True)
    mk = torch.lt(Ev, 0.0)
    Ev[mk] = 0.0 
    I = spectrumInformativeness(Ev,alpha = alpha, variant = variant, n_eig = n_eig)
    return I

def nonInformativeSpectrum(Ev ,average = False, K = None, n_eig = 1):
    """ Finds a non-informative spectrum based on the set of normalized 
    eigen-values of a gram-matrix K.
    Args: 
        Ev: Set of eigen-values from a trace-normalized correlation matrix
        average: Boolean (default False). Chooses weather or not the average of 
                 correlations is used to find the non-informative matrix
        K: (optional) Gram-matrix required in case of using the average
        n_eig: Defines the number of eigen-values to preserve while finding the 
             closest non-informative matrix,
    """
    if average and K is None:
        raise ValueError('If using average, must supply kernel')
   
    n = Ev.shape[0]
    Evni = torch.zeros_like(Ev)
    if average:
        avg = (K.sum()-n) / (n**2-n)
        Evni[0] = 1/n + (n-1)*avg/n
        Evni[1:] = (1-Evni[0])/(n-1)
    else:
        Evni[:n_eig] = Ev[:n_eig]
        Evni[n_eig:] = torch.mean(Ev[n_eig:]) # (1-Ev[0])/(n-1)
    return Evni

def alpha_norm(Ev,alpha):
    """
    This functions compute the alpha-norm to the alpha power for a set
    of normalized eigen-values
    """
    alph_norm = torch.sum(Ev**alpha)
    return alph_norm

def renyiJointInformativeness(Kx,Ky,alpha):
    """
    This functions compute the renyi Joint informativenes for a couple
    of Gram-matrices
    """
    Hj = itl.matrixAlphaJointEntropy([Kx, Ky], alpha=alpha)    
    Nx = closestNonInformative(Kx, average = True)
    Ny = closestNonInformative(Ky, average = True)
    Hni = itl.matrixAlphaJointEntropy([Nx, Ny], alpha=alpha)
    I = Hni - Hj
    return I

def closestNonInformative(K, average = True):
    """
    Computes the "closest" non-informative matrix to 
    a gram-matrix K
    """
    n = K.shape[0]
    if average:
        avg = (K.sum()-n) / (n**2-n)
        diag  = torch.diag(torch.ones(n) - avg)
        off_diag = avg*torch.ones([n,n])
        N = diag + off_diag
    else:
        ev1,_ = torch.lobpcg(K, k=1)
        delta = (n*ev1 -1)/(n-1)
        diag  = torch.diag(torch.ones(n) - delta)
        off_diag = delta*torch.ones([n,n])
        N = diag + off_diag               
    return N

def nonInformativeAlphaEntropy(N, alpha=1.01):
    """
    Computes the entropy for a non-informative matrix:
    
    Args: (N x N) non-informative matrix. All the off-diagonal
          elements must be equal and between 0 and 1. 
    """
    n  = N.shape[0]
    Evni = torch.zeros(n)
    avg = N[0,1]
    Evni[0] = 1/n + (n-1)*avg/n
    Evni[1:] = (1-Evni[0])/(n-1)
    Hni = (1/(1-alpha))*torch.log(alpha_norm(Evni,alpha))
    return Hni


def separability(K,alpha, c = 2, variant = 'renyi'):
    
    """ Computes alpha order separability measurements for 
    correlation matrices. In general separability can be defined as 
    some sort of "distance" between a correlation matrix and its "closest" 
    separable matrix. 
   
    Args:
      K: (N x N) Gram matrix.
      alpha: order of the entropy (norm).
      variant: type of separability. Options: Renyi, ratio, Tsallis, distance
    
    Returns:
      S: Separability. 
    """
    Ev, _ = torch.linalg.eigh(K)
    Ev, _ = Ev.sort(descending = True)
    mk = torch.lt(Ev, 0.0)
    Ev[mk] = 0.0
    n = Ev.shape[0]
    Ev = Ev/torch.sum(Ev)
    S = eigenSeparability(Ev, alpha = alpha, c = c)
    return S

def eigenSeparability(Ev,alpha, variant = 'renyi', c = 2):

    EvSep = separableSpectrum(Ev,c = c)

    if variant == 'renyi':
        H   = (1/(1-alpha))*torch.log(alpha_norm(Ev,alpha))
        Hsep = (1/(1-alpha))*torch.log(alpha_norm(EvSep,alpha))
        S = H - Hsep
    elif variant == 'ratio':
        if alpha < 1:
            S = alpha_norm(Ev,alpha)/alpha_norm(EvSep,alpha)
        else:
            S = alpha_norm(EvSep,alpha)/alpha_norm(Ev,alpha)
    elif variant == 'tsallis':
        q = alpha
        H   = (1/(q-1))*(1-alpha_norm(Ev,alpha))
        Hsep = (1/(q-1))*(1-alpha_norm(EvSep,alpha))
        S = H - Hsep
    elif variant == 'distance':
        S = alpha_norm(EvSep - Ev,alpha) 
    else:
        raise ValueError(" Got a false variant value")
    return S

def separableSpectrum(Ev,c = 2):
    n = Ev.shape[0]
    Evsep = torch.zeros_like(Ev)
    portion = torch.sum(Ev[c:]) / c
    Evsep[:c] = Ev[:c] + portion
    return Evsep