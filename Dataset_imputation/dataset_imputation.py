import os
import torch 
import torch.nn as nn
import numpy as np
import pickle as pkl
from sklearn.mixture import GaussianMixture


def instantiate_DatasetSamplingImputation(param,):
    """ Instantiate a DatasetSamplingImputation object from the param dictionary. """
    return DatasetSamplingImputation(dataset_to_impute = param["dataset_to_impute"], )


class DatasetSamplingImputation(nn.Module):
    def __init__(self, dataset_to_impute, ):
        super().__init__()
        self.dataset_to_impute = dataset_to_impute
        self.loader = None
        self.iter_loader = None
        self.batch_size = None

    def create_loader(self, batch_size, num_workers = 4):
        if batch_size == 1 :
            self.batch_size = 2
        self.loader = torch.utils.data.DataLoader(self.dataset_to_impute, batch_size = batch_size, shuffle = True, num_workers = num_workers)
        self.iter_loader = iter(self.loader)
        self.batch_size = batch_size

    def next_item(self, batch_size, num_workers = 4):
        try :
            current_tensor = next(self.iter_loader)[0]
        except StopIteration :
            self.iter_loader = iter(self.loader)
            current_tensor = next(self.iter_loader)[0]
        while len(current_tensor)<batch_size:
            try :
                current_tensor = torch.cat((current_tensor, next(self.iter_loader)[0]))
            except StopIteration :
                self.iter_loader = iter(self.loader)
                current_tensor = torch.cat((current_tensor, next(self.iter_loader)[0]))
        return current_tensor[:batch_size]        

    def forward(self, x, mask = None, index = None):
        batch_size = x.shape[0]
        if self.loader is None :
            self.create_loader(batch_size, num_workers = 4)
        impute_data = self.next_item(batch_size).to(x.device)
            
        return impute_data

