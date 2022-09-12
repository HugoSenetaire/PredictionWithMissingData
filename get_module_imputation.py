from .LatentImputation import *
from .SklearnImputation import *
from .VAEACImputation import *
from .MixtureOfLogisticsImputation import MixtureOfLogisticsImputation, instantiate_MixtureofLogistics
from .Dataset_imputation import DatasetSamplingImputation, instantiate_DatasetSamplingImputation

list_module_imputation = {
    "VAEACImputation" :VAEACImputation,
    "GaussianMixtureLatentImputation" : GaussianMixtureLatentImputation,
    "GaussianMixtureImputation" : GaussianMixtureImputation,
    "GaussianMixtureDatasetImputation" : GaussianMixtureDatasetImputation,
    "KmeansDatasetImputation" : KmeansDatasetImputation,
    "MixtureOfLogisticsImputation" : MixtureOfLogisticsImputation,
    "DatasetSamplingImputation" : DatasetSamplingImputation,
    "None" : None,
    None : None,
}

list_instantiate_module_imputation = {
    "VAEACImputation" :instantiate_VAEACImputation,
    "GaussianMixtureLatentImputation" : instantiate_GaussianMixtureLatentImputation,
    "GaussianMixtureImputation" : instantiate_GaussianMixtureImputation,
    "GaussianMixtureDatasetImputation" : instantiate_GaussianMixtureDatasetImputation,
    "KmeansDatasetImputation" : instantiate_KmeansDatasetImputation,
    "MixtureOfLogisticsImputation" : instantiate_MixtureofLogistics,
    "DatasetSamplingImputation": instantiate_DatasetSamplingImputation,
}


def get_module_imputation_type(module_imputation_name):
    try :
        return list_module_imputation[module_imputation_name]
    except KeyError :
        raise ValueError(f"Module imputation {module_imputation_name} is not implemented")

def instantiate_module_imputation(module_imputation_name, param):
    if module_imputation_name is None or module_imputation_name == "None" :
            return None
    elif module_imputation_name in list_instantiate_module_imputation :
        return list_instantiate_module_imputation[module_imputation_name](param)
    else :    
        raise ValueError(f"Module imputation {module_imputation_name} instantiation is not implemented")