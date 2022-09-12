from .abstract_imputation import Imputation

import torch
import torch.nn as nn



class SumImpute(Imputation):
  def __init__(self,
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None,
              add_mask = True,
              post_process_regularization = None,
              **kwargs):
      super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                      nb_imputation_mc = nb_imputation_mc,
                      nb_imputation_iwae_test = nb_imputation_iwae_test,
                      nb_imputation_mc_test = nb_imputation_mc_test, 
                      reconstruction_reg=reconstruction_reg, 
                      mask_reg=mask_reg,
                      add_mask= add_mask,
                      post_process_regularization=post_process_regularization,)
      
  def imputation_function(self, data, mask, index = None):
      return torch.sum(data * mask, axis=-1)
   

class ModuleImputation(Imputation):
  def __init__(self, 
              module,
              nb_imputation_iwae = 1,
              nb_imputation_mc = 1,
              nb_imputation_iwae_test = None,
              nb_imputation_mc_test = None,
              reconstruction_reg = None,
              mask_reg = None, 
              add_mask = True,
              post_process_regularization = None,
              **kwargs):
      super().__init__(nb_imputation_iwae = nb_imputation_iwae,
                      nb_imputation_mc = nb_imputation_mc,
                      nb_imputation_iwae_test = nb_imputation_iwae_test,
                      nb_imputation_mc_test = nb_imputation_mc_test,
                      reconstruction_reg=reconstruction_reg, 
                      mask_reg=mask_reg,
                      add_mask= add_mask,
                      post_process_regularization=post_process_regularization,)
      self.module = module
      for param in self.module.parameters():
          param.requires_grad = False
    
  
  def imputation_function(self, data, mask, index = None):
      with torch.no_grad():
        if data.is_cuda and torch.cuda.device_count() > 1:
          self.module.to("cuda:1")
          aux_data = data.to("cuda:1")
          aux_mask = mask.to("cuda:1")
        if index is not None :
          aux_index = index.to("cuda:1")
        else:
          aux_index = None
        imputation = self.module(data, mask, index = index)
      imputation = imputation.to(data.device)
      data_imputed = data * mask + (1-mask) * imputation
      return data_imputed
