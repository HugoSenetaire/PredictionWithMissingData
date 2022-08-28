import torch
import torch.nn as nn
import numpy as np

class Gaussian(nn.Module):
  def __init__(self, dim = 100):
    super(Gaussian, self).__init__()
    self.dim = torch.tensor(np.prod(dim), dtype=torch.int64)
    self.mu = nn.parameter.Parameter(torch.normal(mean = torch.zeros((dim,)), std = torch.ones((dim,))), requires_grad = True)
    self.log_diagonal = nn.parameter.Parameter(torch.normal(mean = torch.zeros((dim,)), std = torch.ones((dim,))), requires_grad = True,)
    self.tril_indices = torch.tril_indices(self.dim, self.dim, -1)
    self.lower_tri = nn.parameter.Parameter(torch.normal(mean = torch.zeros(len(self.tril_indices[0])), std = torch.ones(len(self.tril_indices[0]))), requires_grad = True,)
    self.count = 0

  def get_cov(self):
    aux = torch.zeros((self.dim, self.dim), dtype=torch.float32)
    aux[self.tril_indices[0], self.tril_indices[1]] = self.lower_tri
    aux += torch.diag(torch.exp(self.log_diagonal))
    cov = torch.matmul(aux, aux.t())
    return cov

  def forward(self, x, mask):
    
    cov = self.get_cov()
    log_prob_list = []
    mask = mask.type(torch.bool)

    if torch.all(mask == True):
      p_x = torch.distributions.MultivariateNormal(self.mu, cov)
      return -p_x.log_prob(x), None
    else :
      for i, (current_x, current_mask) in enumerate(zip(x, mask)):
          current_x_s = current_x[current_mask]
          mu_s = self.mu[current_mask]
          cov_s = cov[current_mask][:, current_mask]
          p_x = torch.distributions.MultivariateNormal(mu_s, cov_s) # TODO (@hhjs) : Change this to use LowerTriangular
          log_prob_list.append(-p_x.log_prob(current_x_s))
      return torch.stack(log_prob_list), None

      