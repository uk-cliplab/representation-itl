# Implementations of the difference of entropies equation below, as well as some variations

import torch
import repitl.matrix_itl as itl

def permuteGram(K):
    """
    Randomly permutes the rows and columns of a square matrix
    """
    
    assert K.shape[0] == K.shape[1], f"matrix dimensions must be the same"
    idx = torch.randperm(K.shape[0])
    K = K[idx, :]
    K = K[:, idx]
    return K

def doe(Kx, Ky, alpha, n_iters=10, shouldReturnComponents = False):
    """
    Computes the difference of entropy equation of the following form. Let P be a random permutation matrix
    
    doe(Kx, Ky) = EXPECTATION[ H_alpha(Kx, P Ky P) - H_alpha(Kx, Ky)]
    """
    
    H = itl.matrixAlphaJointEntropy([Kx, Ky], alpha=alpha)
    
    H_perm_avg = 0
    for i in range(n_iters):
        H_perm = itl.matrixAlphaJointEntropy([Kx, permuteGram(Ky)], alpha=alpha)
        H_perm_avg = H_perm_avg + (H_perm / n_iters)
    
    if shouldReturnComponents:
        return H_perm_avg - H, H, H_perm_avg
    
    return H_perm_avg - H

def dip(Kx, Ky, alpha, n_iters=10, allow_exact_compute=True):
    """
    Computes the difference of information potential (DIP) equation of the following form. Let P be a random permutation matrix
    
    doe(Kx, Ky) = EXPECTATION[ GIP_alpha(Kx, P Ky P) - GIP_alpha(Kx, Ky)]
    """
    
    if allow_exact_compute and alpha==2:
        return exact_dip(Kx, Ky)
    
    GIP = itl.generalizedInformationPotential(Kx * Ky, alpha=alpha)
    
    GIP_perm_avg = 0
    for i in range(n_iters):
        GIP_perm = itl.generalizedInformationPotential(Kx * permuteGram(Ky), alpha=alpha)
        GIP_perm_avg = GIP_perm_avg + (GIP_perm / n_iters)

    return GIP_perm_avg - GIP


def exact_dip(Kx, Ky):
    """
    Assuming alpha=2, computes the exact value of dip(Kx, Ky) with no expectation needed
    """

    n = Ky.shape[0]
    Ky_squared_mean = (torch.sum(torch.pow(Ky, 2)) - n) / (n**2 - n)

    Kp = Ky_squared_mean * torch.ones((n,n)) + (1 - Ky_squared_mean) * torch.eye(n)
    
    Ky_normed = (Kp - torch.pow(Ky, 2)).fill_diagonal_(0).type(torch.complex64)
    Ky_normed = torch.pow(Ky_normed, 0.5)
    
    return torch.real(itl.frobeniusGIP(Kx * Ky_normed))

def dip_symmetric(Kx, Ky, alpha, n_iters=10, allow_exact_compute=True):
    return 0.5*dip(Kx, Ky, alpha, n_iters, allow_exact_compute=allow_exact_compute)  + 0.5*dip(Ky, Kx, alpha, n_iters, allow_exact_compute=allow_exact_compute)

def doe_symmetric(Kx, Ky, alpha, n_iters=10):
    return 0.5*doe(Kx, Ky, alpha, n_iters)  + 0.5*doe(Ky, Kx, alpha, n_iters)
