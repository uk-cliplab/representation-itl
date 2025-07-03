'''
random kernel utils
Define randomzed kernel functions
for faster computation
'''

import torch
import numpy as np

# TODO: make this compatible with torch.nn.Module class
# Make sure the seed are outside of this module
# seeds should be set outside on the scripts calling this utilities


class rffGaussian():
    def __init__(self,
                 n_in_dim,
                 n_comp=256,
                 sigma=None,
                 requires_grad=False,
                 ):
        self.n_in_dim = n_in_dim
        self.n_comp = n_comp
        self.requires_grad = requires_grad
        if sigma is None:
            self.sigma = np.sqrt(n_in_dim / 2.0)
        else:
            self.sigma = sigma
        self.sample_weights()

    def __call__(self, x):
        WX = torch.matmul(x, self.W)
        return torch.cat((torch.cos(WX), torch.sin(WX)), axis=1) * self.a

    def sample_weights(self):
        W_ = np.random.normal(size=(self.n_in_dim, self.n_comp // 2))
        W_ /= self.sigma
        self.W = torch.tensor(W_,
                              requires_grad=self.requires_grad,
                              dtype=torch.float32)
        self.a = torch.sqrt(torch.tensor(2.0 / self.n_comp))

    def set_n_comp(self, n_comp):
        if n_comp != self.n_comp:
            self.n_comp = n_comp
            self.sample_weights()

    def set_sigma(self, sigma):
        if sigma != self.sigma:
            self.sigma = sigma
            self.sample_weights()
