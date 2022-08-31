import torch
import torch.nn as nn



class PredictionModule(nn.Module):

    def __init__(self, classifier, imputation = None, imputation_reg = None, input_size = None,**kwargs):
        super(PredictionModule, self).__init__()
        self.classifier = classifier
        self.imputation = imputation
        self.input_size = input_size


    def get_imputation(self, data, mask, index = None):
        """ Get the imputation of the data given the mask
        
        Parameters:
        -----------
        data : torch.Tensor of shape (nb_sample_z_MC * nb_sample_z_iwae * batch_size, channels, size_lists...)
            The data to be imputed
        mask : torch.Tensor of shape (nb_sample_z_MC * nb_sample_z_iwae * batch_size, channels, size_lists...)
            The mask to be used for the imputation, shoudl be in the same shape as the data
        index : torch.Tensor of shape (nb_sample_z_MC * nb_sample_z_iwae * batch_size, )
            The index to be used for imputation
        """
        data = data.reshape(mask.shape) # Quick fix when the reshape function do not match the shape of the data (change the dataset might be better), TODO
        x_imputed, _ = self.imputation(data, mask, index)
        return x_imputed

    def __call__(self, data, mask = None, index = None):
        """ Using the data and the mask, do the imputation and classification 
        
        Parameters:
        -----------
        data : torch.Tensor of shape (nb_sample_z_MC * nb_sample_z_iwae * batch_size, channels, size_lists...)
            The data to be classified
        mask : torch.Tensor of shape (nb_sample_z_MC * nb_sample_z_iwae * batch_size, channels, size_lists...)
            The mask to be used for the classification, shoudl be in the same shape as the data
        index : torch.Tensor of shape (nb_sample_z_MC * nb_sample_z_iwae * batch_size, )
            The index to be used for imputation

        Returns:
        --------
        y_hat : torch.Tensor of shape (nb_imputation * nb_sample_z_MC * nb_sample_z_iwae * batch_size, nb_category)
            The output of the classification
        loss_reconstruction : torch.Tensor of shape (1)
            Some regularization term that can be added to the loss (for instance in the case of version Autoencoder regularisation)
        """
        if mask is not None :
            mask = mask.reshape(data.shape)

        if self.imputation is not None and mask is not None :
            x_imputed, loss_reconstruction = self.imputation(data, mask, index)
            y_hat = self.classifier(x_imputed)
        else :
            y_hat = self.classifier(data)
            loss_reconstruction = torch.zeros((1), device = data.device)

        return y_hat, loss_reconstruction


