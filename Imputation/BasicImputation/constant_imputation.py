from .abstract_imputation import Imputation

import torch
import torch.nn as nn


class ConstantImputation(Imputation):
  def __init__(self,
              cste = 0,
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None,
              add_mask = False,
              post_process_regularization = None,
              **kwargs,
              ):

    super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                    nb_imputation_mc = nb_imputation_mc,
                    nb_imputation_iwae_test = nb_imputation_iwae_test,
                    nb_imputation_mc_test = nb_imputation_mc_test,
                    reconstruction_reg=reconstruction_reg,
                    mask_reg=mask_reg,
                    add_mask= add_mask,
                    post_process_regularization=post_process_regularization,
                    )

    self.cste = nn.parameter.Parameter(torch.tensor(cste), requires_grad=False)

  def has_constant(self):
    return True

  def get_constant(self):
    return self.cste

  def imputation_function(self, data, mask, index = None):
    if torch.any(torch.isnan(data)):
      data_imputed = torch.where(mask==0, torch.full_like(data,torch.tensor(0.)), data)
      data_imputed = mask * data_imputed + (1-mask) * self.cste
    else :
      data_imputed = mask * data + (1-mask) * self.cste
    return data_imputed

class MultipleConstantImputation(Imputation):
  def __init__(self, 
              cste_list_dim = [-2, 2],
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None,
              add_mask = False,
              post_process_regularization = None,
              **kwargs,
            ):

    super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                    nb_imputation_mc = nb_imputation_mc,
                    nb_imputation_iwae_test = nb_imputation_iwae_test,
                    nb_imputation_mc_test = nb_imputation_mc_test,
                    reconstruction_reg=reconstruction_reg,
                    mask_reg=mask_reg,
                    add_mask= add_mask,
                    post_process_regularization=post_process_regularization,
                    )

    self.cste_list_dim = nn.parameter.Parameter(torch.tensor(cste_list_dim), requires_grad=False)

  def has_constant(self):
    return False


  def imputation_function(self, data, mask, index = None):
    if torch.any(torch.isnan(data)):
      data_imputed = torch.where(mask==0, torch.full_like(data,torch.tensor(0.)), data)
    else :
      data_imputed = data
    data_imputed = data_imputed *  mask + (1-mask) * self.cste_list_dim
    return data_imputed



class LearnConstantImputation(Imputation):
  def __init__(self, 
              cste=None,
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None,
              add_mask = False,
              post_process_regularization = None,
              **kwargs,
              ):

    super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                    nb_imputation_mc = nb_imputation_mc,
                    nb_imputation_iwae_test = nb_imputation_iwae_test,
                    nb_imputation_mc_test = nb_imputation_mc_test,
                    reconstruction_reg=reconstruction_reg,
                    mask_reg=mask_reg,
                    add_mask= add_mask,
                    post_process_regularization=post_process_regularization,
                    **kwargs,
                    )

    if cste is None :
      cste = torch.rand(1,).type(torch.float32)
    self.cste = nn.Parameter(cste, requires_grad = True)

  def imputation_function(self, data, mask, index = None):
    if torch.any(torch.isnan(data)):
      data_imputed = torch.where(mask==0, torch.full_like(data,torch.tensor(0.)), data)
    else :
      data_imputed = data
    data_imputed = mask * data_imputed + (1-mask) * self.cste
    # data_imputed = torch.where(mask==0, torch.full_like(data,self.cste), data)
    return data_imputed 
