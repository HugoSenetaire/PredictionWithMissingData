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
   
