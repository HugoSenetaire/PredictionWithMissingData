import torch
import torch.nn as nn





## Different Utils :
def prepare_process(input_process):
  """ Utility function to make sure that the input_process is a list of function or None"""
  if input_process is None:
    return None
  else :
    if input_process is not list :
      input_process = [input_process]
    input_process = nn.ModuleList(input_process)

    return input_process





def expand_for_imputations(data, mask, nb_imputation_iwae, nb_imputation_mc, index = None, collapse = False):
    wanted_reshape = torch.Size((1,)) + torch.Size((data.shape[0],)) + torch.Size((1,)) + data.shape[1:]
    wanted_transform = torch.Size((nb_imputation_mc,)) + torch.Size((data.shape[0],)) + torch.Size((nb_imputation_iwae,)) + data.shape[1:]
    data_expanded = data.reshape(wanted_reshape).expand(wanted_transform)
    mask_expanded = mask.reshape(wanted_reshape).expand(wanted_transform)
    if index is not None :
      wanted_reshape = torch.Size((1,)) + torch.Size((index.shape[0],)) + torch.Size((1,)) + index.shape[1:]
      wanted_transform_index = torch.Size((nb_imputation_iwae,)) + index.shape + torch.Size((nb_imputation_mc,))
      index_expanded = index.reshape(wanted_reshape).expand(wanted_transform_index)
    else:
      index_expanded = None

    if collapse :
      data_expanded = data_expanded.flatten(0, 2)
      mask_expanded = mask_expanded.flatten(0, 2)
      if index is not None :
        index_expanded = index_expanded.flatten(0, 2)
    return data_expanded, mask_expanded, index_expanded

