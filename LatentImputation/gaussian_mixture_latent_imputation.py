import os
import torch 
import torch.nn as nn
import numpy as np
import pickle as pkl
from sklearn.mixture import GaussianMixture



def train_gmm_latent(data, autoencoder_wrapper, n_components, save_path):
    """ Training a Gaussian Mixture Model on the data using sklearn. """
    print("TRAINING FOR {} COMPONENTS".format(n_components))
    if not os.path.exists(os.path.dirname(save_path)):
      os.makedirs(os.path.dirname(save_path))

    if torch.cuda.is_available():
      data = data.cuda()
      autoencoder_wrapper = autoencoder_wrapper.cuda()
    else :
      data = data.cpu()
      autoencoder_wrapper = autoencoder_wrapper.cpu()
    data_masked = autoencoder_wrapper.get_imputation(data, mask = torch.ones_like(data))
    latent_z = autoencoder_wrapper.classifier.encode(data_masked,)
    latent_z = latent_z.detach().numpy()
    gm = GaussianMixture(n_components=n_components, covariance_type='diag',)
    batch_size = latent_z.shape[0]
    latent_flatten = latent_z.reshape(batch_size, -1)
    print(latent_flatten.shape)
    gm.fit(latent_flatten)
    mu = gm.means_
    covariances = gm.covariances_
    weights = gm.weights_
    pkl.dump((weights, mu, covariances), open(save_path, "wb"))
    print("save at ", save_path)
    print(f"{n_components} components saved")




class GaussianMixtureLatentBasedImputation(nn.Module):
  def __init__(self, imputation_network_weights_path, autoencoder_wrapper, **kwargs) :
    super().__init__()
    if not os.path.exists(imputation_network_weights_path):
      raise ValueError("Weights path does not exist for the Gaussian Mixture at {}".format(imputation_network_weights_path))
    with open(imputation_network_weights_path, "rb") as f:
     weights, means, covariances = pkl.load(f)

    
    self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)
    self.means = nn.Parameter(torch.tensor(means, dtype = torch.float32), requires_grad=False)
    self.covariances = nn.Parameter(torch.tensor(covariances, dtype = torch.float32), requires_grad=False)
    self.nb_centers = np.shape(means)[0]

    self.autoencoder_wrapper = autoencoder_wrapper



  def get_dependency(self, data, mask, index = None):
      """ Using the data and the mask, to get the dependency of every point to the component of the mixture
       
        Parameters:
        -----------
        data : torch.Tensor of shape (batch_size, channels, size_lists...)
            The data used for sampling, might have already been treated
        mask : torch.Tensor of shape (batch_size, size_lists...)
            The mask to be used for the classification, shoudl be in the same shape as the data
        index : torch.Tensor of shape (batch_size, size_lists...)
            The index to be used for imputation

        Returns:
        --------
        dependency : torch.Tensor of shape (batch_size, self.nb_centers)
            Get dependency for all data and centers

      """
      with torch.no_grad():
        data_masked = self.autoencoder_wrapper.get_imputation(data, mask, index)
        latent = self.autoencoder_wrapper.classifier.encode(data_masked)
        self.latent_shape = latent.shape

        batch_size = latent.shape[0]
        other_dim = latent.shape[1:]
        self.other_dim_shape = other_dim 

        wanted_shape = torch.Size((batch_size, self.nb_centers, *other_dim))
        wanted_shape_flatten = torch.Size((batch_size, self.nb_centers,np.prod(other_dim),))

        latent_expanded = latent.detach().unsqueeze(1).expand(wanted_shape).reshape(wanted_shape_flatten)
        
        centers = self.means.unsqueeze(0).expand(wanted_shape_flatten)
        variance = self.covariances.unsqueeze(0).expand(wanted_shape_flatten)
        weights = self.weights.unsqueeze(0).expand(torch.Size((batch_size, self.nb_centers,)))


        dependency = -(latent_expanded - centers)**2/2/variance - torch.log(variance)/2
        dependency = torch.sum(dependency,axis=-1) + torch.log(weights)
        dependency_max, _ = torch.max(dependency, axis = -1, keepdim = True)
        dependency -= torch.log(torch.sum(torch.exp(dependency - dependency_max) + 1e-8, axis = -1, keepdim=True)) + dependency_max
        dependency = torch.exp(dependency)

      return dependency, wanted_shape
      
  def __call__(self, data, mask, index=None,):
    raise NotImplementedError

def instantiate_GaussianMixtureLatentImputation(param):
    """ Instantiate a GaussianMixtureLatentImputation object from the param dictionary. """
    imputation_network_weights_path = param["imputation_network_weights_path"]
    autoencoder = param["module"].prediction_module
    return GaussianMixtureLatentImputation(imputation_network_weights_path, autoencoder, )
    



class GaussianMixtureLatentImputation(GaussianMixtureLatentBasedImputation):
  def __init__(self, imputation_network_weights_path, autoencoder_wrapper, **kwargs):
    super().__init__(imputation_network_weights_path, autoencoder_wrapper,)
    

  def __call__(self, data, mask, index = None,):
    """ Using the data and the mask, do the imputation and classification 
        Note : This is just doing a single sample multiple imputation, maybe it might be quicker to allow for multiple imputation ?        
        
        Parameters:
        -----------
        data : torch.Tensor of shape (batch_size, channels, size_lists...)
            The data used for sampling, might have already been treated
        mask : torch.Tensor of shape (batch_size, size_lists...)
            The mask to be used for the classification, shoudl be in the same shape as the data
        index : torch.Tensor of shape (batch_size, size_lists...)
            The index to be used for imputation

        Returns:
        --------
        sampled : torch.Tensor of shape (batch_size, nb_category)
            Sampled tensor from the Gaussian Mixture

        """

    with torch.no_grad() :
      dependency, wanted_shape = self.get_dependency(data, mask, index)
      index_resampling = torch.distributions.Multinomial(probs = dependency).sample().type(torch.int64)
      index_resampling = torch.argmax(index_resampling,axis=-1)
      wanted_centroids = self.means[index_resampling]
      wanted_covariances = self.covariances[index_resampling]

      wanted_shape = data.shape
      sampled_latent = torch.normal(wanted_centroids, torch.sqrt(wanted_covariances)).type(torch.float32).reshape(self.latent_shape)
      sampled = self.autoencoder_wrapper.classifier.decode(sampled_latent).reshape(wanted_shape)

    return sampled
