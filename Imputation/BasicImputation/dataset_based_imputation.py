from .abstract_imputation import Imputation

import torch
import torch.nn as nn

class DatasetBasedImputation(Imputation):
    def __init__(self, 
                dataset,
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
        self.dataset = dataset
        self.exist = hasattr(dataset, "impute") 
        if not self.exist :
          self.nb_imputation = 1
          print(f"There is no theoretical method for multiple imputation with {dataset}. DatasetBasedImputation is bypassed from now on.")
        

    def imputation_function(self, data, mask, index = None):
        if self.exist :
          if self.training :
            dataset_type = "Train"
          else :
            dataset_type = "Test"
          imputed_output = self.dataset.impute(value = data.detach(), mask = mask.detach(), index = index, dataset_type = dataset_type)
          data_imputed = mask * data + (1-mask) * imputed_output
          return data_imputed
        else :
          return data

