import torch.nn as nn




class WrapperAutoencoder(nn.Module):
    def __init__(self, prediction_module):
        super(WrapperAutoencoder, self).__init__()
        self.prediction_module = prediction_module
        self.autoencoder = prediction_module.autoencoder 
        self.prediction_module.imputation.nb_imputation_iwae = 1
        self.prediction_module.imputation.nb_imputation_mc = 1


    def encode(self, x, mask, index):
        data_imputed_autoencoder = self.prediction_module.get_imputation(x, mask, index)
        latent = self.encode(data_imputed_autoencoder)
        return latent

    def decode(self, latent):
        return self.autoencoder.decode(latent)

    