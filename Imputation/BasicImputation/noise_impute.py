from .abstract_imputation import Imputation

import torch
import torch.nn as nn



class NoiseImputation(Imputation):
  def __init__(self,
              sigma = 1.0,
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
    self.sigma = sigma
    assert self.sigma > 0

  def imputation_function(self, data, mask, index = None):
    normal = torch.distributions.normal.Normal(torch.zeros_like(mask), torch.full_like(mask, fill_value= self.sigma))
    noise = normal.sample()
    return data * mask + (1-mask) *  noise 
