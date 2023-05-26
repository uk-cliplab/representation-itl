"""
Kernel utilities
Miscelaneous functions to compute kernels using tensorflow
 
"""
import torch
from typing import TypeVar

## Define Gaussian kernel a la Ermon
Tensor = TypeVar(torch.tensor)

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
    

def gaussianKernel(X, Y, sigma):
    """ Compute the Gram matrix using a Gaussian kernel.
    
    where K(i, j) = exp( - (1/(2 * sigma^2)) * || X[i,::] - Y[j, ::] ||^2 )
    Args:
      X: A tensor with N as the first dim.
      Y: a tensor with M as the first dim.
      sigma: scale parameter (scalar)
             
    
    Returns:
      K: a N x M gram matrix
    """
    D = squaredEuclideanDistanceTiled(X,Y)
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
