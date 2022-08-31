import torch
import torch.nn as nn


class NetworkBasedReconstructionRegularization(nn.Module):
  def __init__(self, network_reconstruction, lambda_reconstruction=0.1,):
    super(NetworkBasedReconstructionRegularization, self).__init__()
    self.network_reconstruction = network_reconstruction
    self.lambda_reconstruction = nn.Parameter(torch.tensor(lambda_reconstruction, dtype = torch.float32), requires_grad = False)

  def __call__(self, data_imputed, data, mask, index = None):
    raise NotImplementedError



### LOSS REGULARIZATION : 

class AutoEncoderLatentReconstruction(NetworkBasedReconstructionRegularization):
  def __init__(self, network_reconstruction, lambda_reconstruction=0.1,):
    super(AutoEncoderLatentReconstruction, self).__init__(network_reconstruction, lambda_reconstruction=lambda_reconstruction,)

  def __call__(self, data_imputed, data, mask, index = None):
    latent_z_imputed = self.network_reconstruction.encode(data_imputed).flatten(1)
    latent_z = self.network_reconstruction.encode(data).flatten(1)

    dim_data = data.flatten(1).shape[1]
    mask_lambda = torch.sum(mask.flatten(1), dim = 1, dtype=torch.float32) / dim_data

    mse_per_batch = torch.sum((latent_z_imputed - latent_z)**2, dim = 1)
    loss = self.lambda_reconstruction * mask_lambda * mse_per_batch
    loss = torch.mean(loss)
    
    return loss

class AutoEncoderReconstructionRegularization(NetworkBasedReconstructionRegularization):
  """
  Add an extra loss to the complete model so that the reconstruction of the imputed image is not too far from the original image.
  """
  def __init__(self, network_reconstruction, lambda_reconstruction = 0.1,):
    super().__init__(network_reconstruction = network_reconstruction, lambda_reconstruction = lambda_reconstruction,)

  
  def __call__(self, data_imputed, data, mask, index = None,):
    data_reconstruced = self.network_reconstruction(data_imputed)
    loss = self.lambda_reconstruction * nn.functional.mse_loss(data_reconstruced, data)
    return loss


reconstruction_regularization_list = {
  "auto_encoder": AutoEncoderReconstructionRegularization,
  "None": None,
  "none": None,
}


def get_reconstruction_regularization(reconstruction_reg,):
  if reconstruction_reg is None:
    return None
  elif reconstruction_reg in reconstruction_regularization_list:
    return reconstruction_regularization_list[reconstruction_reg]
  else:
    raise NotImplementedError