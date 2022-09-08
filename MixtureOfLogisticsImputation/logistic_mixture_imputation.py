import torch.nn as nn
import numpy as np
import pickle as pkl
import torch

class MixtureOfLogisticsImputation(nn.Module):
    def __init__(self, mixture):
        super(MixtureOfLogisticsImputation, self).__init__()
        self.mixture = mixture
    
    def __call__(self, data, mask, index=None,):
        with torch.no_grad():
            return self.mixture.impute(data, mask,)
