from .constant_imputation import LearnConstantImputation, ConstantImputation, MultipleConstantImputation
from .no_imputation import MaskAsInput, NoDestructionImputation
from .module_imputation import ModuleImputation, SumImpute
from .dataset_based_imputation import DatasetBasedImputation
from .noise_impute import NoiseImputation
import torch
import torch.nn as nn





imputation_list = {
  "LearnConstantImputation" : LearnConstantImputation,
  "SumImpute" : SumImpute,
  "ModuleImputation" : ModuleImputation,
  "DatasetBasedImputation" : DatasetBasedImputation,
  "NoiseImputation" : NoiseImputation,
  "ConstantImputation" : ConstantImputation,
  "MultipleConstantImputation" : MultipleConstantImputation,
  "MaskAsInput" : MaskAsInput,
  "NoDestructionImputation" : NoDestructionImputation,
}

## Imputation :
def get_imputation_type(imputation_name):  
  if imputation_name in imputation_list :
    return imputation_list[imputation_name]
  else :
    raise ValueError(f"Imputation {imputation_name} is not implemented")