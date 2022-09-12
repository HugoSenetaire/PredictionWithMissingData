import torch.nn as nn
import numpy as np
import pickle as pkl
import torch

class MixtureOfLogisticsImputation(nn.Module):
    def __init__(self, mixture, mean_imputation = False):
        super(MixtureOfLogisticsImputation, self).__init__()
        self.mixture = mixture
        self.mean_imputation = mean_imputation
    
    def __call__(self, data, mask, index=None,):
        with torch.no_grad():
            return self.mixture.impute(data, mask, mean_sample = self.mean_imputation)
