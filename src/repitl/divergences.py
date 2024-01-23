import torch as torch
import numpy as np

import repitl.kernel_utils as ku
import repitl.matrix_itl as itl
import repitl.matrix_itl_approx as approx

def divergenceJR(X,Y,sigma,alpha, weighted = False, n_rff = None):
    if weighted:
        if n_rff is None:
            divergence = divergenceJR_weighted(X,Y,sigma,alpha)
        else:
            divergence = divergenceJR_weighted_rff(X,Y,sigma,alpha, n_rff)
    else:
        if n_rff is None:
            divergence = divergenceJR_unweighted(X,Y,sigma,alpha)  
        else:
            divergence = divergenceJR_unweighted_rff(X,Y,sigma,alpha,n_rff)
    return divergence

def divergenceJR_weighted(X,Y,sigma,alpha):
    # Getting number of samples from each class
    N = X.shape[0]
    M = Y.shape[0]
    # Creating indicator or label variable
    l = torch.ones(N+M, dtype=torch.long, device=X.device)
    l[N:] = 0
    # One-hot encoding the label variable
    L = torch.nn.functional.one_hot(l).type(X.dtype)
    # Creating the mixture of the two distributions
    XY = torch.cat((X,Y))
    # Kernel of the mixture
    K = ku.gaussianKernel(XY,XY, sigma)
    # Kernel of the labels    
    Kl = torch.matmul(L, L.t())

    # Weight matrices
    D1 = np.sqrt((N+M)/(2*N))*torch.ones(N,1,dtype=X.dtype) # (n1+n2)
    D2 = np.sqrt((N+M)/(2*M))*torch.ones(M,1,dtype=X.dtype)
    D = torch.cat((D1,D2))
    # Weighted kernels
    K_weighted = D*K*torch.t(D)
    Kl_weighted  = D*Kl*torch.t(D)
    
    # Computing divergence
    Hxy = itl.matrixAlphaEntropy(K_weighted, alpha=alpha)
    Hj = itl.matrixAlphaJointEntropy([K_weighted, Kl], alpha=alpha)
    Hl = approx.matrixAlphaEntropyLabel(L, alpha, weighted = True)
    divergence  = Hxy + Hl - Hj
    return divergence

def divergenceJR_unweighted(X,Y,sigma,alpha):
    # Getting number of samples from each class
    N = X.shape[0]
    M = Y.shape[0]
    # Creating indicator or label variable
    l = torch.ones(N+M, dtype=torch.long, device=X.device)
    l[N:] = 0
    # One-hot encoding the label variable
    L = torch.nn.functional.one_hot(l).type(X.dtype)
    # Creating the mixture of the two distributions
    XY = torch.cat((X,Y))
    # Kernel of the mixture
    K = ku.gaussianKernel(XY,XY, sigma)
    # Kernel of the labels  
    Kl = torch.matmul(L, L.t())
    # Computing divergence
    Hxy = itl.matrixAlphaEntropy(K, alpha=alpha)
    Hj = itl.matrixAlphaJointEntropy([K, Kl], alpha=alpha)
    Hl = approx.matrixAlphaEntropyLabel(L, alpha)
    divergence  = Hxy + Hl - Hj
    return divergence

def divergenceJR_weighted_rff(X,Y,sigma,alpha, n_rff):
    # Getting number of samples from each class
    N = X.shape[0]
    M = Y.shape[0]
    # Creating indicator or label variable
    l = torch.ones(N+M, dtype=torch.long, device=X.device)
    l[N:] = 0
    # One-hot encoding the label variable
    L = torch.nn.functional.one_hot(l).type(X.dtype)
    # Creating the mixture of the two distributions
    XY = torch.cat((X,Y))
    # Approximating covariance matrices
    D1 = np.sqrt((N+M)/(2*N))*torch.ones(N,1,dtype=X.dtype) # (n1+n2)
    D2 = np.sqrt((N+M)/(2*M))*torch.ones(M,1,dtype=X.dtype)
    D = torch.cat((D1,D2))
    cov_x   = approx.cov_matrix_rff_weighted(X ,sigma,n_rff,D1,random = False)
    cov_y   = approx.cov_matrix_rff_weighted(Y ,sigma,n_rff,D2,random = False)
    cov_xy  = approx.cov_matrix_rff_weighted(XY,sigma,n_rff,D ,random = False)

    Hxy = itl.matrixAlphaEntropy(cov_xy, alpha=alpha)
    # Then, to approximate the joint entropy K॰Kl we use the properties of block diagonal matrices, 
    # where the trace of the entire matrix is just the sum of the trace of the 2 blocks in the matrix
    Hj = approx.KzoKl_entropy_rff_weighted(cov_x,cov_y,alpha,sigma)
    # Entropy of the label or indicator variable
    Hl = approx.matrixAlphaEntropyLabel(L, alpha)
    # Finally, we compute the divergence
    divergence = Hxy + Hl - Hj
    return divergence

def divergenceJR_unweighted_rff(X,Y,sigma,alpha, n_rff):
    # Getting number of samples from each class
    N = X.shape[0]
    M = Y.shape[0]
    # Creating indicator or label variable
    l = torch.ones(N+M, dtype=torch.long, device=X.device)
    l[N:] = 0
    # One-hot encoding the label variable
    L = torch.nn.functional.one_hot(l).type(X.dtype)
    # Creating the mixture of the two distributions
    XY = torch.cat((X,Y))
    # Approximating covariance matrices
    cov_x = approx.cov_matrix_rff(X,sigma,n_rff,random = False)
    cov_xy = approx.cov_matrix_rff(XY,sigma,n_rff,random = False)
    Hxy = itl.matrixAlphaEntropy(cov_xy, alpha=alpha)
    # Then, to approximate the joint entropy K॰Kl we use the properties of block diagonal matrices, 
    # where the trace of the entire matrix is just the sum of the trace of the 2 blocks in the matrix
    # cov_x was computed previously
    cov_y = approx.cov_matrix_rff(Y,sigma,n_rff,random = False)
    Hj = approx.KzoKl_entropy_rff(cov_x,cov_y,alpha,sigma)
    # Entropy of the label or indicator variable
    Hl = approx.matrixAlphaEntropyLabel(L, alpha)
    # Finally, we compute the divergence
    divergence = Hxy + Hl - Hj
    return divergence

def repJSD(X,Y, weighted = False):
    phiX = X
    phiY = Y
    N = phiX.shape[0]
    M = phiY.shape[0]
    if weighted:
        pix = 0.5
        piy = 0.5
    else:
        pix = N/(N+M)
        piy = M/(N+M)
    # Creating the mixture of both distributions
    # phiZ =  torch.cat((phiX,phiY))
    covX = 1/N*(phiX.T @ phiX)
    covY = 1/M*(phiY.T @ phiY)
    Hx = itl.vonNeumannEntropy(covX)
    Hy = itl.vonNeumannEntropy(covY)
    Hz = itl.vonNeumannEntropy(pix*covX+piy*covY)
    JSD =  (Hz - (pix*Hx + piy*Hy))
    return JSD

def repJSD_cov(covX,covY, pix = 0.5, piy = 0.5):
    Hx = itl.vonNeumannEntropy(covX)
    Hy = itl.vonNeumannEntropy(covY)
    Hz = itl.vonNeumannEntropy(pix*covX+piy*covY)
    JSD =  (Hz - (pix*Hx + piy*Hy))
    return JSD

def repJSD_approx(X,Y, weighted = False, typeApprox = 'radial'):
    phiX = X
    phiY = Y
    N = phiX.shape[0]
    M = phiY.shape[0]
    if weighted:
        pix = 0.5
        piy = 0.5
    else:
        pix = N/(N+M)
        piy = M/(N+M)
    # Creating the mixture of both distributions
    # phiZ =  torch.cat((phiX,phiY))
    covX = 1/N*(phiX.T @ phiX)
    covY = 1/M*(phiY.T @ phiY)

    Hx = approx.fastVonNeumannEntropy(covX,typeApprox)
    Hy = approx.fastVonNeumannEntropy(covY,typeApprox)
    Hz = approx.fastVonNeumannEntropy(pix*covX+piy*covY,typeApprox)
    JSD =  (Hz - (pix*Hx + piy*Hy))
    return JSD

def repJRD(X,Y,alpha):
    phiX = X
    phiY = Y
    N = phiX.shape[0]
    M = phiY.shape[0]
    pix = N/(N+M)
    piy = M/(N+M)
    # rewrite covX and covY using @ instead of matmul and.T instead of t()
    covX = 1/N*(phiX.T @ phiX)
    covY = 1/M*(phiY.T @ phiY)
    Hx = itl.matrixAlphaEntropy(covX, alpha=alpha)
    Hy = itl.matrixAlphaEntropy(covY, alpha=alpha)
    Hz = itl.matrixAlphaEntropy((pix*covX+piy*covY), alpha=alpha)
    JRD =  (Hz - (pix*Hx + piy*Hy))
    return JRD

def repJRD_cov(covX,covY,alpha,pix= 0.5,piy= 0.5):
    Hx = itl.matrixAlphaEntropy(covX, alpha=alpha)
    Hy = itl.matrixAlphaEntropy(covY, alpha=alpha)
    Hz = itl.matrixAlphaEntropy((pix*covX+piy*covY), alpha=alpha)
    JRD =  (Hz - (pix*Hx + piy*Hy))
    return JRD

def repKL(X,Y):
    # X and Y are outputs of a Fourier Feature Layer. They should be the same size every vector should be norm 1
    Cx = (1/X.shape[0])*torch.matmul(torch.t(X),X)
    Cy = (1/Y.shape[0])*torch.matmul(torch.t(Y),Y)
    Lx, Qx = torch.linalg.eigh(Cx)
    Ly, Qy = torch.linalg.eigh(Cy)
    Lx[Lx<=0] = 1e-10 # this is to avoid nans in the log 
    Ly[Ly<=0] = 1e-10    
    logLx = torch.log(Lx)
    logLy = torch.log(Ly)
    logCx = torch.matmul(Qx,torch.matmul(torch.diag(logLx),Qx.t())) 
    logCy = torch.matmul(Qy,torch.matmul(torch.diag(logLy),Qy.t())) 
    return torch.trace(torch.matmul(Cx,logCx - logCy))

def repKL_cov(Cx,Cy):
    Lx, Qx = torch.linalg.eigh(Cx)
    Ly, Qy = torch.linalg.eigh(Cy)
    Lx[Lx<=0] = 1e-10 # this is to avoid nans in the log 
    Ly[Ly<=0] = 1e-10   
    logLx = torch.log(Lx)
    logLy = torch.log(Ly)
    logCx = torch.matmul(Qx,torch.matmul(torch.diag(logLx),Qx.t())) 
    logCy = torch.matmul(Qy,torch.matmul(torch.diag(logLy),Qy.t())) 
    return torch.trace(torch.matmul(Cx,logCx - logCy)) 
