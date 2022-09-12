import os
import torch 
import torch.nn as nn
import numpy as np
import pickle as pkl
from sklearn.mixture import GaussianMixture


def instantiate_DatasetSamplingImputation(param,):
    """ Instantiate a DatasetSamplingImputation object from the param dictionary. """
    return DatasetSamplingImputation(data_to_impute = param["data_to_impute"], )


class DatasetSamplingImputation(nn.Module):
    def __init__(self, data_to_impute, ):
        super().__init__()
        self.data_to_impute = data_to_impute


    def forward(self, x, mask = None, index = None):
        batch_size = x.shape[0]
        impute_index = np.random.choice(np.arange(len(self.data_to_impute)), batch_size)
        impute_data = self.data_to_impute[impute_index].to(x.device)
        return impute_data

