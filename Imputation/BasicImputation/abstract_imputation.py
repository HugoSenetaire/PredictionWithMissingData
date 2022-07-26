import torch
import torch.nn as nn
from utils import prepare_process, expand_for_imputations


## Abstract imputation class :

class Imputation(nn.Module):
  def __init__(self,
            nb_imputation_iwae = 1,
            nb_imputation_mc = 1,
            nb_imputation_iwae_test = None,
            nb_imputation_mc_test = None,
            reconstruction_reg = None,
            mask_reg = None,
            add_mask = False,
            post_process_regularization = None,
            **kwargs):
    super().__init__()
    self.add_mask = add_mask
    self.reconstruction_reg = prepare_process(reconstruction_reg)
    self.post_process_regularization = prepare_process(post_process_regularization)
    self.mask_reg = prepare_process(mask_reg)

    self.nb_imputation_iwae = nb_imputation_iwae
    if nb_imputation_iwae_test is None :
      self.nb_imputation_iwae_test = 1
    else :
      self.nb_imputation_iwae_test = nb_imputation_iwae_test

    self.nb_imputation_mc = nb_imputation_mc
    if nb_imputation_mc_test is None :
      self.nb_imputation_mc_test = 1
    else :
      self.nb_imputation_mc_test = nb_imputation_mc_test
  
  def has_constant(self):
    return False
 
  def has_rate(self):
    return False



  def reconstruction_regularization(self, data_imputed, data, mask, index = None):
    loss_reconstruction = torch.tensor(0., device = data.device)
    if self.reconstruction_reg is not None :
      for process in self.reconstruction_reg :
          loss_reconstruction += process(data_imputed, data, mask, index = index)
    return loss_reconstruction

  
  def add_mask_method(self, data_imputed, mask):
    if len(mask.shape)>2:
      mask_aux = mask[:,0].unsqueeze(1)
    else :
      mask_aux = mask
    return torch.cat([data_imputed, mask_aux], axis =1)


  def post_process(self, data_imputed, data, mask, index = None):
    if self.post_process_regularization is not None :
      for process in self.post_process_regularization:
        data_imputed, mask = process(data_imputed, data, mask, index = index)
    if self.add_mask:
      data_imputed = self.add_mask_method(data_imputed, mask)
    return data_imputed

  def mask_regularization(self, data, mask):
    if self.mask_reg is None :
      return mask
    else :
      for process in self.mask_reg :
        mask = process(data, mask)
      return mask


  def imputation_function(self, data, mask, index=None):
    raise NotImplementedError

  def forward(self, data, mask, index = None):
    mask = self.mask_regularization(data, mask)

    if self.training :
      nb_imputation_mc = self.nb_imputation_mc
      nb_imputation_iwae = self.nb_imputation_iwae
    else :
      nb_imputation_mc = self.nb_imputation_mc_test
      nb_imputation_iwae = self.nb_imputation_iwae_test

    data_expanded, mask_expanded, index_expanded = expand_for_imputations(data, mask, nb_imputation_mc = nb_imputation_mc, nb_imputation_iwae=nb_imputation_iwae, index = index, collapse = True)
    data_imputed = self.imputation_function(data_expanded, mask_expanded, index_expanded)
    loss_reconstruction = self.reconstruction_regularization(data_imputed, data_expanded, mask_expanded, index = index_expanded)
    data_imputed = self.post_process(data_imputed, data_expanded, mask_expanded, index = index_expanded)
    return data_imputed, loss_reconstruction


