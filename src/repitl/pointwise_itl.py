"""pointwise information theoretic quantities for itl.

Define our core pointwise quantity, the pointwise vonNeumann
entropy and relateed quantities
"""

import torch


# Auxiliary function to disregard any egnevalues below epsilon precision
def zlog(x, eps=None):
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    idx = torch.argwhere(x > eps)
    y = torch.zeros_like(x)
    y[idx] = torch.log(x[idx]/torch.sum(x[idx]))
    return y


def pointwiseVNEntropy(Kxm, Km):
    D, V = torch.linalg.eig(Km / torch.trace(Km))
    D = torch.abs(D)
    phi = torch.matmul(Kxm, V.real / torch.sqrt(D * D.shape[0])[None, :])
    H = -torch.matmul(torch.pow(phi, 2), zlog(D))
    return H


def pointwiseMI(Kxm, Kym, Km, kernels=None):
    assert len(Km) == 2, 'Km should be a list or tuple with two models'
    Km_x = Km[0]
    Km_y = Km[1]
    Hx = pointwiseVNEntropy(Kxm, Km_x)
    Hy = pointwiseVNEntropy(Kym, Km_y)
    Hxy = pointwiseVNEntropy(Kxm * Kym, Km_x * Km_y)
    return Hx + Hy - Hxy
