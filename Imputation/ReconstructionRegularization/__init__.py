from .reconstruction_loss_regularization import AutoEncoderReconstructionRegularization, AutoEncoderLatentReconstruction, get_reconstruction_regularization

dic_reconstruction_regularization = {
    "AutoEncoderReconstructionRegularization" : AutoEncoderReconstructionRegularization,
    "AutoEncoderLatentReconstruction" : AutoEncoderLatentReconstruction,
    "None" : None,
    "none" : None,
}