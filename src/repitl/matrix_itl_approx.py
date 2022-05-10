import numpy as np
import torch
from functools import reduce

def Z_rff(X,sigma,n_rff,random = True):
    '''
    Function to compute Random Fourier Features to approximate
    the mapping induced by a Gaussian Kernel. 
    Args:
        X: set of points for computing covariance matrix
        sigma: Kernel bandwidth
        n_rff:  Number of Random Fourier Features to compute
        random: Parameter to control the generation of random fourier features
    Returns:
        Z_rff: Random Fourier Features
    '''
    if not random:
        # Set seed to generate the same set of random weights
        torch.manual_seed(5)
    n_rff = torch.tensor(n_rff)
    gamma = torch.tensor(1/(2*sigma**2))
    # Number of features in X
    d = X.shape[1]
    # Random Fourier features
    w = torch.sqrt(2*gamma)*torch.randn(d, n_rff,dtype=X.dtype)
    b = 2*np.pi*torch.rand(n_rff,dtype=X.dtype)
    Z = torch.sqrt(2/n_rff)*torch.cos(torch.matmul(X,w) + b)
    return Z

def cov_matrix_rff(X,sigma,n_rff,random = True):
    '''
    Function to approximate the covariance matrix (via Random Fourier Features)
    of the fatures induced by a Gaussian Kernel
    Args:
        X: set of points for computing covariance matrix
        sigma: Kernel bandwidth
        n_rff:  Number of Random Fourier Features to compute
        random: Parameter to control the generation of random fourier features
    Returns:
        cov_matrix: Covariance matrix
    '''    
    Z = Z_rff(X,sigma,n_rff,random = random)
    # Covariance matrix
    cov_matrix = torch.matmul(torch.t(Z),Z)
    return cov_matrix / Z.shape[0] 

def cov_matrix_rff_weighted(X,sigma,n_rff,D,random = True):
    """
    Function to approximate the covariance matrix (via Random Fourier Features)
    of the fatures induced by a Gaussian Kernel in a weighted form.
    Args:
        X: set of points for computing covariance matrix
        sigma: Kernel bandwidth
        n_rff:  Number of Random Fourier Features to compute
        D: Weight matrix 
        random: Parameter to control the generation of random fourier features
    Returns:
        cov_matrix: Weighted covariance matrix
    """
    Z  = Z_rff(X,sigma,n_rff,random = random)
    Z_ = D*Z 
    cov_matrix = torch.matmul(torch.t(Z_),Z_)
    return cov_matrix / Z_.shape[0]

def joint_cov_matrix_rff(X,Y,sigma,D):
    XY = torch.cat((X,Y),1)
    joint_cov_matrix = cov_matrix_rff(XY,sigma = sigma,D = D)
    return joint_cov_matrix
 
def KzoKl_entropy_rff(cov_x,cov_y,alpha,sigma):
    '''
    This function approximates the computation of 
    the entropy of Kz hadamard Kl (KzoKl) from the 
    covariance matrices of X and Y
    Z = [X;Y]
    Args:
        cov_x,cov_y: Covariances of X and Y
        alpha: order of the entropy
        sigma: Kernel bandwidth
    Returns: 
        Hj = Alpha order entropy approximation of KzoKl

    '''
    ex, _ = torch.symeig(cov_x, eigenvectors=True)  
    mx = torch.gt(ex, 0.0)
    mex = ex[mx]
    
    ey, _ = torch.symeig(cov_y, eigenvectors=True)  
    my = torch.gt(ey, 0.0)
    mey = ey[my]
    
    mexy = torch.cat((mex,mey))
    mexy = mexy / (torch.sum(mex)+torch.sum(mey))
    
    GIP = torch.sum(torch.exp(alpha * torch.log(mexy))) 
    Hj = (1.0 / (1.0 - alpha)) * torch.log(GIP)
    return Hj

def KzoKl_entropy_rff_weighted(cov_x,cov_y,alpha,sigma):
    ex, _ = torch.symeig(cov_x, eigenvectors=True)  
    mx = torch.gt(ex, 0.0)
    mex = ex[mx]
    mex /= 2*torch.sum(mex)
    
    ey, _ = torch.symeig(cov_y, eigenvectors=True)  
    my = torch.gt(ey, 0.0)
    mey = ey[my]
    mey /= 2*torch.sum(mey)
    
    mexy = torch.cat((mex,mey))
    
    GIP = torch.sum(torch.exp(alpha * torch.log(mexy))) 
    Hj = (1.0 / (1.0 - alpha)) * torch.log(GIP)
    return Hj

def matrixAlphaEntropyLabel(L, alpha, weighted = False):
    '''
    This function compute the entropy of the indicator variable
    L, without calculating the eigenvalues, which are constant 
    according to the distribution of the classes.
    Args: 
        L: Indicator varibale one-hot-encoded
        alpha: order of the entropy
        weighted: parameter to control the weight given to each class
    Returns: 
        H: Alpha order entropy 
    '''
    if weighted:
        exy = torch.tensor([0.5,0.5])
    else:
        Nx = torch.sum(L[:,0])
        Ny = torch.sum(L[:,1])
        exy = torch.tensor([Nx/(Nx+Ny),Ny/(Nx+Ny)])
    
    GIP = torch.sum(torch.exp(alpha * torch.log(exy)))
    H = (1.0 / (1.0 - alpha)) * torch.log(GIP)
<<<<<<< HEAD
    return H        
=======
    return H
        
def joint_entropy_rff(cov_x,cov_y,alpha,sigma,n_rff):    
    ex, _ = torch.symeig(cov_x, eigenvectors=True)  
    mx = torch.gt(ex, 0.0)
    mex = ex[mx]
    
    ey, _ = torch.symeig(cov_y, eigenvectors=True)  
    my = torch.gt(ey, 0.0)
    mey = ey[my]
    
    mexy = torch.cat((mex,mey))
    mexy = mexy / (torch.sum(mex)+torch.sum(mey))
    
    GIP = torch.sum(torch.exp(alpha * torch.log(mexy))) 
    Hj = (1.0 / (1.0 - alpha)) * torch.log(GIP)
    return Hj

def joint_entropy_rff_norm(cov_x,cov_y,alpha,sigma):
    ex, _ = torch.symeig(cov_x, eigenvectors=True)  
    mx = torch.gt(ex, 0.0)
    mex = ex[mx]
    mex /= 2*torch.sum(mex)
    
    ey, _ = torch.symeig(cov_y, eigenvectors=True)  
    my = torch.gt(ey, 0.0)
    mey = ey[my]
    mey /= 2*torch.sum(mey)
    
    mexy = torch.cat((mex,mey))
    
    GIP = torch.sum(torch.exp(alpha * torch.log(mexy))) 
    Hj = (1.0 / (1.0 - alpha)) * torch.log(GIP)
    return Hj
>>>>>>> 9895151ac7fd683db2523369fe6250e5c0f81717

def generalizedInformationPotential(K, alpha):
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
    ek, _ = torch.symeig(K, eigenvectors=True)  
    mk = torch.gt(ek, 0.0)
    mek = ek[mk]
    mek = mek / torch.sum(mek)
    GIP = torch.sum(torch.exp(alpha * torch.log(mek))) 
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
