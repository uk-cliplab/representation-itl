"""
Kernel utilities
Miscelaneous functions to compute kernels using tensorflow
 
"""
import torch
from typing import TypeVar

## Define Gaussian kernel a la Ermon
Tensor = torch.tensor

def manhattanDistanceTiled(x: Tensor, y: Tensor) -> Tensor:
    """ Compute matrix with pairwise Manhattan distances 
    between the rows of X and the rows of Y
                 
                 D_{i,j} = ||x_i - y_j ||_1
            
    Args:
      X: A tensor with N as the first dim.
      Y: a tensor with M as the first dim.
      the remaining dimensions of X and Y must match
    Returns:
      D: a N x M array where 
    """
    N = x.shape[0] 
    M = y.shape[0]
    dim = x.shape[1]
    x = x.unsqueeze(1) # (N, 1, dim)
    y = y.unsqueeze(0) # (1, M, dim)
    tiled_x = x.expand(N, M, dim)
    tiled_y = y.expand(N, M, dim)
    return torch.abs((tiled_x - tiled_y)).sum(2)


def ellipticalLaplacianKernel(X, Y, sigma):
    """ Compute the Gram matrix using a Gaussian kernel.
    
    where K(i, j) = exp( - (1/ sigma) * || X[i,::] - Y[j, ::] ||_2 )
    Args:
      X: A tensor with N as the first dim.
      Y: a tensor with M as the first dim.
      sigma: scale parameter (scalar)
             
    
    Returns:
      K: a N x M gram matrix
    """
    D = squaredEuclideanDistance(X,Y)
    return  torch.exp( -torch.sqrt(D / 2) / sigma)

def factorizedLaplacianKernel(X, Y, sigma):
    """ Compute the Gram matrix using a Gaussian kernel.
    
    where K(i, j) = exp( - (1/ sigma) * || X[i,::] - Y[j, ::] ||_1 )
    Args:
      X: A tensor with N as the first dim.
      Y: a tensor with M as the first dim.
      sigma: scale parameter (scalar)
             
    
    Returns:
      K: a N x M gram matrix
    """
    D = manhattanDistanceTiled(X,Y)
    return  torch.exp( -D / (torch.sqrt(torch.tensor(2, dtype=D.dtype)) * sigma))


def softmaxKernel(X, Y):
    Sx = torch.nn.functional.softmax(X, dim=-1)
    Sx = Sx / torch.sqrt(torch.sum(torch.square(Sx), axis=1, keepdims=True))
    Sy = torch.nn.functional.softmax(Y, dim=-1)
    Sy = Sy / torch.sqrt(torch.sum(torch.square(Sy), axis=1, keepdims=True))
    return torch.matmul(Sx, Sy.t())

def exponentialKernel(X, Y, sigma):
    return torch.exp( X @ Y.T / sigma)

def linearKernel(X, Y, sigma):
    return (X @ Y.T) / sigma
 
def squaredEuclideanDistanceTiled(x: Tensor,
                                  y: Tensor) -> Tensor:
    """ Compute matrix with pairwise squared Euclidean distances 
    between the rows of X and the rows of Y
                 
                 D_{i,j} = ||x_i - y_j ||^2

    Args:
      X: A tensor with N as the first dim.
      Y: a tensor with M as the first dim.
      the remaining dimensions of X and Y must match
    Returns:
      D: a N x M array where 
    """
    N = x.shape[0] 
    M = y.shape[0]
    dim = x.shape[1]
    x = x.unsqueeze(1) # (N, 1, dim)
    y = y.unsqueeze(0) # (1, M, dim)
    tiled_x = x.expand(N, M, dim)
    tiled_y = y.expand(N, M, dim)
    return (tiled_x - tiled_y).pow(2).sum(2)

def innerProductTiled(x: Tensor, y: Tensor) -> Tensor:
    """ Compute matrix with pairwise squared Euclidean distances 
    between the rows of X and the rows of Y
                 
                 D_{i,j} = ||x_i - y_j ||^2

    Args:
      X: A tensor with N as the first dim.
      Y: a tensor with M as the first dim.
      the remaining dimensions of X and Y must match
    Returns:
      D: a N x M array where 
    """
    N = x.shape[0] 
    M = y.shape[0]
    dim = x.shape[1]
    x = x.unsqueeze(1) # (N, 1, dim)
    y = y.unsqueeze(0) # (1, M, dim)
    tiled_x = x.expand(N, M, dim)
    tiled_y = y.expand(N, M, dim)
    return (tiled_x * tiled_y).sum(2)

def innerProductNormalizedTiled(x: Tensor, y: Tensor) -> Tensor:
    """ Compute matrix with pairwise squared Euclidean distances 
    between the rows of X and the rows of Y
                 
                 D_{i,j} = ||x_i - y_j ||^2

    Args:
      X: A tensor with N as the first dim.
      Y: a tensor with M as the first dim.
      the remaining dimensions of X and Y must match
    Returns:
      D: a N x M array where 
    """
    N = x.shape[0] 
    M = y.shape[0]
    dim = x.shape[1]
    x = x.unsqueeze(1) # (N, 1, dim)
    y = y.unsqueeze(0) # (1, M, dim)
    tiled_x = x.expand(N, M, dim)
    tiled_y = y.expand(N, M, dim)
    norm_x = tiled_x.pow(2).sum(2).sqrt()
    norm_y = tiled_y.pow(2).sum(2).sqrt()
    return (tiled_x * tiled_y / (norm_x * norm_y)).sum(2)


## Define Gaussian kernel a la Luichi


def squaredEuclideanDistance(X, Y):
    """ Compute matrix with pairwise squared Euclidean distances 
    between the rows of X and the rows of Y
                 
                 D_{i,j} = ||x_i - y_j ||^2

    Args:
      X: A tensor with N as the first dim.
      Y: a tensor with M as the first dim.
      the remaining dimensions of X and Y must match
    Returns:
      D: a N x M array where 
    """
    G12 = torch.matmul(X, Y.t())
    G11 = torch.sum(torch.square(X), axis=1, keepdim=True)
    G22 = torch.sum(torch.square(Y), axis=1, keepdim=True)
    D = G11 - G12 * torch.tensor(2, dtype=X.dtype, device=X.device) + G22.t()
    return D
    

def gaussianKernel(X, Y, sigma=None, calcOption='tiled'):
    """ Compute the Gram matrix using a Gaussian kernel.
    
    where K(i, j) = exp( - (1/(2 * sigma^2)) * || X[i,::] - Y[j, ::] ||^2 )
    Args:
      X: A tensor with N as the first dim.
      Y: a tensor with M as the first dim.
      sigma: scale parameter (scalar)
             
    
    Returns:
      K: a N x M gram matrix
    """
    if calcOption == 'tiled':
        D = squaredEuclideanDistanceTiled(X, Y)
    else:
        D = squaredEuclideanDistance(X, Y)
    if sigma is None:
        sigma = torch.sqrt(torch.tensor(X.shape[1]/2))
        
    return  torch.exp( -D / (2.0 * sigma**2))

# Add diagonal elements of matrix to have multiple backend compatibility
     
def normalizeGramMatrix(G):
    """ Normalizes Gram matrix based on its diagonal
    Matrix must be symmetric

    
    Args:
      G: A symmetric N x N positive definite matrix
    

    Returns:
      G_hat: Normalized N x N matrix with ones in the diagonal
    """
    G_diag = torch.sqrt(torch.diag(G))
    G_hat = G / G_diag[:, None] / G_diag[None, :]
    return G_hat
