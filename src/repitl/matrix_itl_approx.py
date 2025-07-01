import numpy as np
import torch

"""
BEGIN RFF FUNCTIONS
"""


def Z_rff(X,
          sigma,
          n_rff,
          random=True
          ):
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
    n_rff = torch.tensor(n_rff, device=X.device)
    gamma = torch.tensor(1/(2*sigma**2), device=X.device)
    # Number of features in X
    d = X.shape[1]
    # Random Fourier features
    w = torch.sqrt(2*gamma)*torch.randn(d, n_rff,
                                        dtype=X.dtype,
                                        device=X.device)
    b = 2*np.pi*torch.rand(n_rff, dtype=X.dtype, device=X.device)
    Z = torch.sqrt(2/n_rff)*torch.cos(torch.matmul(X, w) + b)
    return Z


def cov_matrix_rff(X,
                   sigma,
                   n_rff,
                   random=True
                   ):
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
    Z = Z_rff(X, sigma, n_rff, random=random)
    # Covariance matrix
    cov_matrix = torch.matmul(torch.t(Z), Z)
    return cov_matrix / Z.shape[0]


def cov_matrix_rff_weighted(X,
                            sigma,
                            n_rff,
                            D,
                            random=True
                            ):
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
    Z = Z_rff(X, sigma, n_rff, random=random)
    Z_ = D*Z
    cov_matrix = torch.matmul(torch.t(Z_), Z_)
    return cov_matrix / Z_.shape[0]


def joint_cov_matrix_rff(X,
                         Y,
                         sigma,
                         D
                         ):
    XY = torch.cat((X, Y), 1)
    joint_cov_matrix = cov_matrix_rff(XY, sigma=sigma, D=D)
    return joint_cov_matrix


def KzoKl_entropy_rff(cov_x,
                      cov_y,
                      alpha,
                      sigma
                      ):
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

    mexy = torch.cat((mex, mey))
    mexy = mexy / (torch.sum(mex)+torch.sum(mey))

    GIP = torch.sum(torch.exp(alpha * torch.log(mexy)))
    Hj = (1.0 / (1.0 - alpha)) * torch.log(GIP)
    return Hj


def KzoKl_entropy_rff_weighted(cov_x,
                               cov_y,
                               alpha,
                               sigma
                               ):
    ex, _ = torch.symeig(cov_x, eigenvectors=True)
    mx = torch.gt(ex, 0.0)
    mex = ex[mx]
    mex /= 2*torch.sum(mex)

    ey, _ = torch.symeig(cov_y, eigenvectors=True)
    my = torch.gt(ey, 0.0)
    mey = ey[my]
    mey /= 2*torch.sum(mey)

    mexy = torch.cat((mex, mey))

    GIP = torch.sum(torch.exp(alpha * torch.log(mexy)))
    Hj = (1.0 / (1.0 - alpha)) * torch.log(GIP)
    return Hj


def matrixAlphaEntropyLabel(L,
                            alpha,
                            weighted=False
                            ):
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
        exy = torch.tensor([0.5, 0.5])
    else:
        Nx = torch.sum(L[:, 0])
        Ny = torch.sum(L[:, 1])
        exy = torch.tensor([Nx / (Nx + Ny), Ny / (Nx + Ny)])

    GIP = torch.sum(torch.exp(alpha * torch.log(exy)))
    H = (1.0 / (1.0 - alpha)) * torch.log(GIP)
    return H


"""
BEGIN ENTROPY UTIL FUNCTIONS
"""


def upperBoundEntropy(ek, N, alpha, lower_distribution='ignore'):
    """
    Computes the entropy from a set of eigenvalues.
    If lower distribution is not ignore, an upper bound it computed

    Args:
        ek: A (perhaps partial) set of eigenvalues
        N: The cardinality of the full set of eigenvalues
        alpha: What alpha to use for the norm
        lower_distribution: how to approximate the remaining
        (K.shape[0] - M) eigenvalues

    Returns:
        An upper bound of entropy based on the eigenvalues
    """
    if lower_distribution not in ['uniform',
                                  'normal',
                                  'exponential',
                                  'ignore']:
        raise ValueError("Invalid choice of lower ev distribution!")

    mk = torch.gt(ek, 0.0)
    mek = ek[mk]

    if torch.sum(mek) < 1 and lower_distribution != 'ignore':
        n_unknown = N - len(mek)

        if lower_distribution == 'uniform':
            lower_ev = ((1 - torch.sum(mek)) / (n_unknown)) * torch.ones(n_unknown)
        elif lower_distribution == 'normal':
            lower_ev = torch.abs(torch.empty(n_unknown).normal_(mean=0, std=0.1))
            lower_ev /= torch.sum(lower_ev)
            lower_ev *= 1 - torch.sum(mek)

        elif lower_distribution == 'exponential':
            lower_ev = torch.abs(torch.empty(n_unknown).exponential_(lambd=2))
            lower_ev /= torch.sum(lower_ev)
            lower_ev *= 1 - torch.sum(mek)

        full_spectrum = torch.cat((mek, lower_ev), 0)
        GIP = torch.sum(torch.exp(alpha * torch.log(full_spectrum)))

        return (1.0 / (1.0 - alpha)) * torch.log(GIP)

    else:
        mek = mek / torch.sum(mek)
        GIP = torch.sum(torch.exp(alpha * torch.log(mek)))
        return (1.0 / (1.0 - alpha)) * torch.log(GIP)


"""
BEGIN SANGER RULE FUNCTIONS
"""


def computeSangersRule(X, M, compute_eigenvalues=True, lr= 5e-8, tolerance=1e-6, max_iters=5e3, initial_weights=None):
    """
    Performs hebbian learning using Sanger's rule
    
    Args:
        X: data matrix
        M: number of eigenvectors to compute
    
    Returns:
        eu, ev: eigenvalues and orthonormal eigenvectors
    """

    # Weight initialization
    if initial_weights is not None:
        W_sanger = initial_weights
        prev_W_sanger = initial_weights
    else:
        W_sanger = torch.rand(size=(X.shape[1], M), dtype=X.dtype) - 0.5
        prev_W_sanger = torch.ones((X.shape[1], M), dtype=X.dtype)

    cur_iter = 0
    while cur_iter < max_iters:
        cur_iter += 1
        prev_W_sanger = W_sanger.detach().clone()
        
        Y = X @ W_sanger
        Q = torch.tril(Y.T @ Y)
        W_sanger_grad = lr * (Y.T @ X - (W_sanger @ Q).T).T
        
        W_sanger = W_sanger + W_sanger_grad

    if compute_eigenvalues:
        cov = X.T @ X
        cov /= torch.trace(cov)
    
        eigenvalues = torch.norm(cov@W_sanger, dim=0) / torch.norm(W_sanger, dim=0)
        eigenvalues, _ = eigenvalues.sort()

        return eigenvalues, W_sanger.detach()
    
    return W_sanger.detach()



def sangerRuleEntropy(K, M, alpha, lr= 5e-3, tolerance=1e-6, max_iters=5e3, initial_weights=None, lower_distribution='ignore'):
    """
    Function to compute an estimate of entropy using sanger's rule
    
    Args:
        K: A square matrix
        M: The number of eigenvectors to approximate.
        lr: The learning rate
        tolerance: The covergence tolerance stopping criterion
        max_iters: maximum number of iterations
        lower_distribution: how to approximate the remaining (K.shape[0] - M) eigenvalues
        
    Returns:
        H: An estimate of the representation entropy of K
        W_sanger:  The resulting approximated eigenvectors
    """
    if M > K.shape[0]:
        raise ValueError("M must be less than or equal to K.shape[0]!")

    Kev, W_sanger = computeSangersRule(K, M, compute_eigenvalues=True,  lr=lr, tolerance=tolerance, max_iters=max_iters, initial_weights=initial_weights)
    
    H = upperBoundEntropy(Kev, K.shape[0], alpha, lower_distribution)

    return H, W_sanger

def sangerRuleMutualInformation(Kx, Ky, M, alpha, lr= 5e-8, tolerance=1e-6, max_iters=5e3, lower_distribution='ignore'):
    Hx = sangerRuleEntropy(Kx, M, alpha, lr=lr, tolerance=tolerance, max_iters=max_iters, lower_distribution=lower_distribution)
    Hy = sangerRuleEntropy(Ky, M, alpha, lr=lr, tolerance=tolerance, max_iters=max_iters, lower_distribution=lower_distribution)
    Hxy = sangerRuleEntropy(Kx * Ky, M, alpha, lr=lr, tolerance=tolerance, max_iters=max_iters, lower_distribution=lower_distribution)
    minf = Hx + Hy - Hxy
        
    return minf


"""
BEGIN EIGENGAME FUNCTIONS
"""
class eigenGamePCA():
    def __init__(self, n_in_dim, n_comp=None, lr=0.01, max_iter=100):
        self.n_in_dim = n_in_dim
        self.n_comp = n_comp
        self.lr = lr
        self.max_iter = max_iter
        V_ = np.random.normal(size=(self.n_in_dim, n_comp))
        V_, _ = np.linalg.qr(V_)
        self.V = torch.tensor(V_, requires_grad=False, dtype=torch.float)
    # forward method
    def fit(self, x):
        # No stopping criteria yet
        for i in range(self.max_iter):
            self.update(x)
        
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def update(self, x):
        XV = self.transform(x)
        rewards = XV 
        C = torch.matmul(XV.T, XV) 
        C = C / torch.diag(C, 0).view(self.n_comp, 1)
        penalties = torch.matmul(XV, torch.triu(C, 1))
        nablaV = 2*torch.matmul(x.T, rewards - penalties)
        nablaVR = nablaV - torch.sum(nablaV * self.V, axis=1, keepdims=True) * self.V
        self.V = self.V + self.lr * nablaVR / XV.shape[0]
        self.V = self.V / torch.sqrt(torch.sum(torch.square(self.V), axis=0, keepdims=True))
        
    def transform(self, x):
        result = torch.matmul(x, self.V)
        self.V = self.V.detach()
        
        return result
    
class rffGaussian():
    def __init__(self, n_in_dim, n_comp=None, sigma=1.0):
        self.n_in_dim = n_in_dim
        if n_comp is None:
            self.n_comp = 1024
        else:
            self.n_comp = n_comp
        
        self.sigma = sigma

    def __call__(self, x):
        self.W = torch.randn(size=(self.n_in_dim, self.n_comp // 2), requires_grad=True)
        self.W = self.W * self.sigma
        
        WX = torch.matmul(x, self.W)
        self.W.detach()
        return torch.cat((torch.cos(WX), torch.sin(WX)), axis=1) * np.sqrt(2 / self.n_comp)

class eigenGameEntropy():
    def __init__(self, n_in_dim, alpha=1.01, n_comp=None, lr=0.01, beta=1.0, max_iter=100):
        self.n_in_dim = n_in_dim
        if n_comp is None:
            self.n_comp = n_in_dim
        else:
            self.n_comp = n_comp
        self.alpha = alpha
        self.lr = lr
        self.beta = beta
        self.max_iter = max_iter
        self.pca = eigenGamePCA(self.n_in_dim, 
                                self.n_comp,
                                self.lr,
                                self.max_iter)
        
        self.eigvals = torch.ones((self.n_comp), dtype=torch.float) / self.n_comp
        
    # forward method
    def fit(self, x):
        z = self.pca.fit_transform(x)
           
    def update(self, x):
        self.pca.update(x)
        z = self.pca.transform(x)
        z_var = torch.mean(torch.square(z), axis=0)
        z_var = z_var / torch.sum(z_var) 
        self.eigvals = (1 - self.beta) * self.eigvals + self.beta * z_var

        return self.entropy()

    def entropy(self):
        gip = torch.sum(torch.pow(self.eigvals, self.alpha))
        
        self.eigvals = self.eigvals.detach()
        
        return torch.log(gip) / (1 - self.alpha)
    

## Quadratic approximations of von neumann entropy

def fastVonNeumannEntropy(K, typeApprox = 'radial'):
    """Computes the von Neumann entropy of a kernel matrix K according to https://arxiv.org/pdf/1811.11087.pdf
    usinf radial projections """
    n = K.shape[0]
    K = (1/torch.trace(K))*K # make sure the trace is one (normalized kernel)
    frob_norm = torch.linalg.matrix_norm(K, ord='fro')

    if typeApprox == 'radial':
        kg = torch.sqrt(frob_norm**2 - (1/n))
        lambda1 = np.sqrt((n-1)/n)*kg + (1/n)
        lambda2 = 1/n - kg/(np.sqrt(n*(n-1))) # this eigenvalue is repeated (n-1) times
        H = -lambda1*torch.log(lambda1) - (n-1)*lambda2*torch.log(lambda2)
    if typeApprox == 'finger':
        lambdamax  = torch.lobpcg(K, k=1, largest=True, method='ortho')[0]
        H = - (torch.log(lambdamax))* (1-frob_norm**2)
    return H
