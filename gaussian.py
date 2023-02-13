import torch
import torch.nn as nn
import numpy as np

class Gaussian(nn.Module):
  def __init__(self, dim = 100, cov = None, init_dataset = None):
    super(Gaussian, self).__init__()
    self.dim = torch.tensor(np.prod(dim), dtype=torch.int64)
    if init_dataset is not None:
      x_train = torch.stack([init_dataset[k]['data'] for k in range(len(init_dataset)) if (init_dataset[k]['mask'].flatten().sum(-1) == self.dim)])
      print(torch.mean(x_train.flatten(1), dim=0))
      self.mu = nn.parameter.Parameter(torch.mean(x_train.flatten(1), dim=0), requires_grad = True)
    else :
      self.mu = nn.parameter.Parameter(torch.normal(mean = torch.zeros((dim,)), std = torch.ones((dim,))), requires_grad = True)

    
    # self.upper_tri = nn.parameter.Parameter(torch.normal(mean = torch.zeros((dim, dim)), std = torch.ones((dim, dim))), requires_grad = True)
    if cov is not None:
      self.cov = torch.tensor(cov, dtype=torch.float32)
    else :
      self.cov = None
      self.log_diagonal = nn.parameter.Parameter(torch.normal(mean = torch.zeros((dim,)), std = torch.ones((dim,))), requires_grad = True,)
      self.tril_indices = torch.tril_indices(self.dim, self.dim, -1)
      self.lower_tri = nn.parameter.Parameter(torch.normal(mean = torch.zeros(len(self.tril_indices[0])), std = torch.ones(len(self.tril_indices[0]))), requires_grad = True,)
      self.count = 0

  def get_cov(self):
    if self.cov is not None:
      return self.cov
    else :
      aux = torch.zeros((self.dim, self.dim), dtype=torch.float32)
      aux[self.tril_indices[0], self.tril_indices[1]] = self.lower_tri
      aux += torch.diag(torch.exp(self.log_diagonal))
      cov = torch.matmul(aux, aux.t())
      # aux = torch.triu(self.upper_tri)
      # cov = torch.matmul(aux.t(), aux)
      return cov

  def forward(self, x, mask):
    
    cov = self.get_cov()
    # cov+=torch.randn(cov.shape)*1e-5
    log_prob_list = []
    mask = mask.type(torch.bool)

    if torch.all(mask == True):
      p_x = torch.distributions.MultivariateNormal(self.mu, cov)
      return -p_x.log_prob(x.flatten(1)), None
    else :
      for i, (current_x, current_mask) in enumerate(zip(x, mask)):
          current_mask_flatten = current_mask.flatten()
          current_x_s = current_x.flatten()[current_mask_flatten]
          mu_s = self.mu[current_mask_flatten]
          cov_s = cov[current_mask_flatten][:, current_mask_flatten]
          p_x = torch.distributions.MultivariateNormal(mu_s, cov_s) # TODO (@hhjs) : Change this to use LowerTriangular
          log_prob_list.append(-p_x.log_prob(current_x_s))
      return torch.stack(log_prob_list), None

      
class GaussianV2(Gaussian):
  def __init__(self, dim = 100, cov = None, init_dataset = None):
    super().__init__(dim = dim, cov = cov, init_dataset = init_dataset)
    if self.cov is not None :
      self.precision = torch.inverse(self.cov)

  def get_precision(self):

    if self.cov is None :
      aux = torch.zeros((self.dim, self.dim), dtype=torch.float32)
      aux[self.tril_indices[0], self.tril_indices[1]] = self.lower_tri
      aux += torch.diag(torch.exp(self.log_diagonal))
      precision = torch.matmul(aux.t(), aux)
    else :
      precision = self.precision
    return precision

  def get_cov(self):
    if self.cov is None :
      precision = self.get_precision()
      cov = torch.inverse(precision) # Should consider the pseudo inverse here ?
    else :
      cov = self.cov
    return cov

  def forward(self, x, mask):
    # cov = self.get_cov()
    precision = self.get_precision()
    # try :
    singular_values_inv = torch.linalg.svdvals(precision) # Rather than calculate this, one could just maximize ||sigma x|| / ||x||
    singular_values = (1/singular_values_inv)+1e-8
    mask = mask.type(torch.bool)

    precision = precision.unsqueeze(0).expand(x.shape[0], -1, -1)
    mu = self.mu.unsqueeze(0).expand(x.shape[0], -1)
    singular_values = singular_values.unsqueeze(0).expand(x.shape[0], -1)
    current_mask_flatten = mask.flatten(1)


    x_s = (x.flatten(1) - mu)*current_mask_flatten
    log_p =  ((x_s.unsqueeze(1) @ precision) @ x_s.unsqueeze(-1)).squeeze()
    log_p = log_p + (torch.log(singular_values.abs())*current_mask_flatten).sum(1)
    log_p += current_mask_flatten.sum(1)*torch.log(torch.tensor(2*np.pi, dtype=torch.float32))
    log_p = log_p/2
    return log_p, None
