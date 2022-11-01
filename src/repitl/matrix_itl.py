"""
matrix itl
Define our core quantities. Matrix Renyi's alpha entropy
and Matrix Renyi's alpha divergence
"""

import torch
from functools import reduce

def generalizedInformationPotential(K, alpha, allow_frobenius_speedup=False):
    """Computes the generalized information 
    potential of order alpha
          GIP_alpha(K) = trace(K_^alpha), 
    where K^alpha is a matrix raised to the alpha power. 
    K_ is normalized as K_ = K / trace(K), such that 
          trace(K_) = 1.
   
    Args:
      K: (N x N) Gram matrix.
      alpha: order of the entropy.
    
    Returns:
      GIP: generalized information potential of alpha order. 
    """
    if allow_frobenius_speedup and alpha == 2:
        return frobeniusGIP(K)
    
    ek, _ = torch.linalg.eigh(K)  
    mk = torch.gt(ek, 0.0)
    mek = ek[mk]
    mek = mek / torch.sum(mek)
    GIP = torch.sum(torch.exp(alpha * torch.log(mek)))
    return GIP


def frobeniusGIP(K):
    """
    calculates entropy using the frobenius norm trick
    equivalent result to calling generalizedInformationPotential(K, alpha=2), but this is much faster

    todo: due to symmetricity of K, can be twice as fast be only considering the lower triangle
    
    Args:
        K: (N x N) Gram matrix
    """
    GIP = torch.sum(torch.pow(K, 2))
    
    # normalize so that the sum of eigenvalues is 1
    GIP /= K.shape[0]**2
    
    return GIP
    
def matrixAlphaEntropy(K, alpha):
    """Computes the matrix based alpha-entropy
    based on the spectrum of K
        H_alpha(A) = (1/(1-alpha))log(trace(A^alpha)), 
    where A^alpha is the matrix power of alpha (A is normalized).
    
    
    Args:
      A: (N x N) Gram matrix.
      alpha: order of the entropy.
    
    Returns:
      H: alpha entropy 
    """
    ##ccompute generalized information Potential    
    GIP = generalizedInformationPotential(K, alpha)
    H = (1.0 / (1.0 - alpha)) * torch.log(GIP)
    return H
    
    
def matrixAlphaJointEntropy(K_list, alpha):
    """Computes the matrix based alpha joint-entropy
    based on the spectrum of K
        H_alpha(K) = (1/(1-alpha))log(trace(K^alpha)), 
    where K^alpha is the matrix power of K (K is normalized).
    
    
    Args:
      K_list: a list of (N x N) Gram matrices.
      alpha: order of the entropy.
      normalize: Boolean (default True)
    
    Returns:
      H: alpha joint entropy 
    """
    K = reduce(lambda x, y: x * y, K_list)
    return matrixAlphaEntropy(K, alpha)


def matrixAlphaConditionalEntropy(Kx, Ky, alpha):
    
    Kxy = Kx * Ky 
    Hxy = matrixAlphaEntropy(Kxy, alpha=alpha)
    Hy = matrixAlphaEntropy(Ky, alpha=alpha)
    return Hxy - Hy

def matrixAlphaMutualInformation(Kx, Ky, alpha):
    Kxy = Kx * Ky 
    Hxy = matrixAlphaEntropy(Kxy, alpha=alpha)
    Hx = matrixAlphaEntropy(Kx, alpha=alpha)
    Hy = matrixAlphaEntropy(Ky, alpha=alpha)
    return Hx + Hy - Hxy
    

def schattenNorm(T, p):
    """Computes the p power of Schatten p-norm of a matrix T
        (|| T ||_p)^p
    """
    _, s, _ = torch.svd(T, some=True)
    return torch.sum(s.pow(p))


def normalizeTriplet(Kx, Ky, Kxy):
    Kxy = Kxy / torch.sqrt(torch.trace(Kx) * torch.trace(Ky))
    Kx = Kx / torch.trace(Kx)
    Ky = Ky / torch.trace(Ky)
    return Kx, Ky, Kxy
    
def schattenDivergence(Kx, Ky, Kxy, p, normalize=False):
    pass

def schatten1Divergence(Kx, Ky, Kxy, normalize=False):
    if normalize:
        Kx, Ky, Kxy = normalizeTriplet(Kx, Ky, Kxy)
    
    D = 1 - schattenNorm(Kxy, p=1.0)
    return D

def matrixAlphaDivergence(Kx, Ky, Kxy, alpha, normalize=False):
    """Computes the matrix based alpha-divergence
        H_alpha(K) = (1/(1-alpha))log(trace(K^alpha)), 
    where K^alpha is the matrix power of K (K is normalized).


    Args:
      Ax: (N x N) Gram matrix.
      Ay: (N x N) Gram matrix.
      Axy: (N x N) Gram matrix.
      alpha : order of the entropy.
      normalize: Boolean (default False)

    Returns:
      D: alpha divergence computed based on the spectrum of Kx, Ky, Kxy
    
    """
    if normalize:
        Kx, Ky, Kxy = normalizeTriplet(Kx, Ky, Kxy)
    
    ex, vx = torch.linalg.eigh(Kx)
    ey, vy = torch.linalg.eigh(Ky)
    # need to correct for negaitve eigenvalues
    mx = torch.gt(ex, 0.0)
    my = torch.gt(ey, 0.0)
    mex = ex[mx]
    mey = ey[my]
    mvx = vx[:, mx]
    mvy = vy[:, my]
    mex = mex / torch.sum(mex)
    mey = mey / torch.sum(mey)
    M = torch.square(torch.matmul(mvx.t(), torch.matmul(Kxy, mvy)))
    B = torch.matmul(torch.pow(mex, (alpha - 1))[None, :], torch.matmul(M, torch.pow(mey, -alpha )[:,None]))
    D = torch.log(B) / (alpha - 1)
    return D


