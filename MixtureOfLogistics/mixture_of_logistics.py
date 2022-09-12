import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Uniform
from .utils import log_prob_from_logits, log_sum_exp, LogisticDistribution



class MixtureOfLogistics(nn.Module):
    def __init__(self, input_size, nb_centers, transform_mean = None, transform_std = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_channels = input_size[0]
        self.nb_centers = nb_centers
        self.log_weight = torch.normal(torch.zeros((self.nb_centers,)), std = torch.ones((self.nb_centers,)))
        self.log_weight = nn.parameter.Parameter(self.log_weight, requires_grad=True)

        self.size_parameters = torch.Size((nb_centers,))+ self.input_size
        self.log_s_parameters = torch.normal(torch.zeros(self.size_parameters), std = torch.ones(self.size_parameters))
        self.log_s_parameters = nn.parameter.Parameter(self.log_s_parameters,requires_grad=True) # initialization is quite poor

        self.mu_parameters = Uniform(-1,1).sample(self.size_parameters)
        self.mu_parameters = nn.parameter.Parameter(self.mu_parameters, requires_grad=True) # initialization is quite poor
        self.transform_mean = transform_mean
        self.transform_std = transform_std 

    def transform_data(self, x):
        if self.transform_mean is not None and self.transform_std is not None:
            aux = (x * self.transform_std) + self.transform_mean
            aux = (aux - 127.5) * 2./255.
        else :
            raise ValueError("You need to provide a transform_mean and transform_std to the MixtureOfLogistics")
      
        return aux

    def log_prob_given_z(self, x, mask = None,):
        """
        Given x and mask, compute the log probability of the missing data per center of the mixture
        """
        transform_x = self.transform_data(x)

        current_log_s = self.log_s_parameters.unsqueeze(1).expand(self.nb_centers, *x.shape)
        current_mu = self.mu_parameters.unsqueeze(1).expand(self.nb_centers, *x.shape)
        current_x = transform_x.unsqueeze(0).expand(self.nb_centers, *x.shape) 
        centered_x = current_x - current_mu

        inv_stdv = torch.exp(-current_log_s)

        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = torch.sigmoid(min_in)
        # log probability for edge case of 0 (before scaling)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        # log probability for edge case of 255 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases
        mid_in = inv_stdv * centered_x
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in our code)
        log_pdf_mid = mid_in - current_log_s - 2.0 * F.softplus(mid_in)

        inner_inner_cond = (cdf_delta > 1e-5).float()
        inner_inner_out = inner_inner_cond * torch.log(
            torch.clamp(cdf_delta, min=1e-12)
        ) + (1.0 - inner_inner_cond) * (log_pdf_mid - np.log(127.5))

        inner_cond = (current_x > 0.999).float()
        inner_out = (
            inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
        )

        cond = (current_x < -0.999).float()
        log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
        if mask is not None and not torch.all(mask==1):
            current_mask = mask.unsqueeze(0).expand(self.nb_centers, *x.shape)
            log_probs = log_probs * current_mask

        log_probs = torch.sum(log_probs, dim=2) # Sum over the channels
        log_probs = torch.sum(log_probs.flatten(2), dim=2) # Sum over the width and height
        return log_probs

    def forward(self, x, mask = None, dependency = None):
        log_probs = self.log_prob_given_z(x, mask = mask)
        if dependency is not None :
            # This can be improved by using dependency in the log_prob_given_z function, and calculate only for the wanted mixture
            if dependency.shape != log_probs.shape :
                dependency = torch.nn.functional.one_hot(dependency, num_classes = self.nb_centers).transpose(1,0)
                assert dependency.shape == log_probs.shape, "The dependency shape is not the same as the log_probs shape"
            log_probs = torch.logsumexp(log_probs * dependency, dim=0)
        else :
            current_log_weights = log_prob_from_logits(self.log_weight, axis = 0)
            current_log_weights = current_log_weights.unsqueeze(1).expand(self.nb_centers, x.shape[0])
            log_probs = log_probs + current_log_weights
            log_probs = torch.logsumexp(log_probs, dim=0) # Sum over the dependency
        return log_probs

        
        
    def get_dependency(self, x, mask = None):
        log_probs = self.log_prob_given_z(x, mask)
        current_log_weights = log_prob_from_logits(self.log_weight, axis = 0)
        current_log_weights = current_log_weights.unsqueeze(1).expand(self.nb_centers, x.shape[0])


        log_probs = log_probs + current_log_weights
        log_dependency = log_prob_from_logits(log_probs, axis = 0)
        dependency = torch.exp(log_dependency)
        return dependency

    def sample(self, x, mask, nb_samples = 1, mean_sample = False):
        if x is None :
            dependency = torch.exp(self.log_weight.unsqueeze(1).expand(self.nb_centers, nb_samples))
        else :
            dependency = self.get_dependency(x, mask)
        index_resampling = torch.distributions.Multinomial(probs = dependency.transpose(0,1)).sample().type(torch.int64)
        index_resampling = torch.argmax(index_resampling,axis=-1)
        
        wanted_s= self.log_s_parameters[index_resampling]
        wanted_mu = self.mu_parameters[index_resampling]
        if mean_sample :
            samples = ((wanted_mu * 255/2 + 127.5) - self.transform_mean)/self.transform_std
        else :
            samples = torch.distributions.Uniform(1e-5* torch.ones_like(wanted_mu), torch.ones_like(wanted_mu) - 1e-5).sample()
            samples = wanted_mu + torch.exp(wanted_s) * (torch.log(samples + 1e-8) - torch.log(1.0 - samples + 1e-8)) 
            samples = torch.clamp(samples, min = -1.0, max = 1.0)
            samples = ((samples * 255/2 + 127.5) - self.transform_mean)/self.transform_std
        
        return samples

