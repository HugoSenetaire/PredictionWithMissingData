import os
import torch 
import torch.nn as nn
import numpy as np
import pickle as pkl
from ..MixtureOfLogistics import MixtureOfLogistics
from torch.utils.data import Dataset, DataLoader
from .logistic_mixture_imputation import MixtureOfLogisticsImputation
import tqdm




def instantiate_MixtureofLogistics(param):
    """ Instantiate a VAEACImputation object from the param dictionary. """
    mixture = MixtureOfLogistics(input_size=param["input_size"], nb_centers = param["nb_component"], transform_mean=param["transform_mean"], transform_std=param["transform_std"])
    path_weights = os.path.join(param["model_dir"], "mixture_of_logistics.pt")
    mixture.load_state_dict(torch.load(path_weights, map_location=next(mixture.parameters()).device))
    mixture_imputation = MixtureOfLogisticsImputation(mixture, mean_imputation = param["mean_imputation"])
    return mixture_imputation

