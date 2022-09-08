import os
import torch 
import torch.nn as nn
import numpy as np
import pickle as pkl
from ..mixture_of_logistics import MixtureOfLogistics
from torch.utils.data import Dataset, DataLoader
from .logistic_mixture_imputation import MixtureOfLogisticsImputation
import tqdm


class DatasetInput(Dataset):
    """CelebA dataset."""

    def __init__(self, data):
        """
        Args:
            data (torch.tensor):       torch.tensor of shape (N, C, H, W)
          
        """
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        return image

class NoTargetDataset(Dataset):
    def __init__(self, dataset, lenght = None):
        self.dataset = dataset
        self.lenght = lenght
        if lenght is not None and lenght< len(self.dataset):
            self.index = np.random.choice(len(self.dataset), lenght, replace=False)

    def __getitem__(self, index):
        if self.lenght is not None :
            new_index = self.index[index]
            return self.dataset[new_index][0]
        else :
            return self.dataset[index][0]

    def __len__(self):
        if self.lenght is not None :
            return self.lenght
        else :
            return len(self.dataset)

def log_prob_from_logits(x, axis= 1):
    """numerically stable log_softmax implementation that prevents overflow""" #Looks like he is softmaxing on the channel distribution
    # TF ordering
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def validation(mixture, k, epochs, dataloader, verbose = True):
    with torch.no_grad():
        complete_loss = torch.tensor(0, device = next(mixture.parameters()).device, dtype = torch.float32)
        for i, data in enumerate(dataloader):
            data = data.to(next(mixture.parameters()).device)
            log_prob = mixture.forward(data)
            complete_loss += -log_prob.sum()
        complete_loss = complete_loss/len(dataloader.dataset)
        repartition_centers = calculate_repartition_centers(mixture, dataloader)
        print("Validation Epoch {} / {}".format(k, epochs))
        print("Validation Loss : {}".format(complete_loss.item()))
        print("Repartition centers : {}".format(repartition_centers))
def sgd_training(epochs, mixture, train_dataset, validation_dataset, batch_size, lr=1e-4, verbose = True):
    optimizer = torch.optim.Adam(mixture.parameters(), lr=lr)

    # build dataloaders on top of datasets
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Validation
    validation(mixture, -1, epochs, val_dataloader)
    for k in range(epochs):
        print("=========================================")
        print("Epoch {}".format(k))
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            data = data.to(next(mixture.parameters()).device)
            log_prob = mixture.forward(data)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose and i % max((len(dataloader)//10),1) == 0:
                print("Batch {} / {}".format(i, len(dataloader)))
                print("Loss : {}".format(loss.item()))
        
        # Validation
        validation(mixture, k, epochs, val_dataloader)

def calculate_repartition_centers(mixture, dataloader, nb_e_step = None):
    """ Calculate the repartition of the centers of the mixture of logistics. """
    if nb_e_step is None:
        nb_e_step = len(dataloader)
    repartition_centers = torch.zeros(mixture.nb_centers, device = next(mixture.parameters()).device, dtype = torch.float32)
    for i, data in tqdm.tqdm(enumerate(dataloader)):
        if i >= nb_e_step:
            break        
        data = data.to(next(mixture.parameters()).device)
        dependency = mixture.get_dependency(data)
        dependency_index = torch.argmax(dependency, dim = 0)
        repartition_centers +=torch.nn.functional.one_hot(dependency_index, num_classes = mixture.nb_centers).sum(0)
    return repartition_centers

def m_step(mixture, dataloader, optimizer, nb_m_step = None):
    print("M step")
    optimizer.zero_grad()
    if nb_m_step is None:
        nb_m_step = len(dataloader)
    # else :
        # nb_m_step = min(nb_m_step, len(dataloader))


    for k in tqdm.tqdm(range(nb_m_step)):
        try :
            data = next(dataloader)
        except StopIteration:
            dataloader = iter(dataloader)
            data = next(dataloader)
        data = data.to(next(mixture.parameters()).device)
        dependency = mixture.get_dependency(data)
        dependency_index = torch.argmax(dependency, dim = 0)
        dependency = torch.nn.functional.one_hot(dependency_index, num_classes = mixture.nb_centers).transpose(0,1)

        log_prob = mixture.forward(data, mask = None, dependency = dependency)
        loss = -log_prob.mean()
        loss.backward()
        optimizer.step()

def e_step(mixture, dataloader, nb_e_step = None):
    print("E step")
    repartition_centers = mixture.nb_centers
    if nb_e_step is None:
        nb_e_step = len(dataloader)

    for k in tqdm.tqdm(range(nb_e_step)):
        try :
            data = next(dataloader)
        except StopIteration :
            data =iter(dataloader)
            data = next(dataloader)
        data = data.to(next(mixture.parameters()).device)
        dependency = mixture.get_dependency(data)
        dependency_index = torch.argmax(dependency, dim = 0)
        repartition_centers +=torch.nn.functional.one_hot(dependency_index, num_classes = mixture.nb_centers).sum(0)
    mixture.log_weight.data = torch.log(repartition_centers)
    mixture.log_weight.data = log_prob_from_logits(mixture.log_weight.data, axis = 0)
    

def em_training(mixture, train_dataset, validation_dataset, nb_m_step = 1, nb_e_step = None, nb_loop = 20, verbose = True, lr = 1e-4, batch_size = 64):
    dataloader = iter(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0))
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mixture.log_weight.requires_grad = False
    optimizer = torch.optim.Adam(mixture.parameters(), lr=lr)
    validation(mixture, -1, nb_loop, val_dataloader, verbose = verbose)
    for k in range(nb_loop):
        print("=========================================")
        print("Loop {}".format(k))
        e_step(mixture = mixture, dataloader = dataloader, nb_e_step = nb_e_step)
        m_step(mixture = mixture, dataloader = dataloader, nb_m_step = nb_m_step, optimizer= optimizer)
        validation(mixture, k, nb_loop, val_dataloader, verbose = verbose,)
        


def train_MixtureOfLogistics(loader,
                            model_dir,
                            nb_component=20,
                            epochs = 20,
                            transform_mean = 0.5,
                            transform_std = 0.5,
                            batch_size = 64,
                            lr = 1e-7,
                            type_of_training = "sgd",
                            nb_m_step = 100,
                            nb_e_step = None,
                            verbose = True):
    """ Training a Gaussian Mixture Model on the data using sklearn. """
    print("TRAINING FROM MODEL DIR {}".format(model_dir))
    use_cuda = torch.cuda.is_available()
    print("Learning rate : {}".format(lr))
    print("type of training : {}".format(type_of_training))
    

    # load train and validation datasets
    try :
        train_dataset = DatasetInput(loader.dataset.data_train)
        index = np.random.choice(range(len(loader.dataset.data_test)), min(1000,len(loader.dataset.data_test)), replace=False)
        validation_dataset = DatasetInput(loader.dataset.data_test[index])
    except AttributeError:
        train_dataset = NoTargetDataset(loader.dataset_train)
        validation_dataset = NoTargetDataset(loader.dataset_test, lenght=1000)

    data_example = next(iter(train_dataset))
    mixture = MixtureOfLogistics(input_size=data_example.shape, nb_centers = nb_component, transform_mean=transform_mean, transform_std=transform_std)
    if use_cuda :
        mixture = mixture.to("cuda:0")

    if type_of_training == "sgd":
        sgd_training(epochs = epochs,
                    mixture = mixture,
                    train_dataset = train_dataset,
                    validation_dataset = validation_dataset,
                    lr = lr,
                    batch_size = batch_size,
                    verbose = verbose)

    elif type_of_training == "em":
        em_training(mixture = mixture,
                    train_dataset = train_dataset,
                    validation_dataset = validation_dataset,
                    batch_size = batch_size,
                    lr=lr,
                    nb_e_step=nb_e_step,
                    nb_m_step=nb_m_step,
                    nb_loop=epochs,
                    verbose = verbose)
    else :
        raise ValueError("type_of_training should be either sgd or em")
    
    # Save the model
    torch.save(mixture.state_dict(), os.path.join(model_dir, f"mixture_of_logistics.pt"))
    


def instantiate_MixtureofLogistics(param):
    """ Instantiate a VAEACImputation object from the param dictionary. """
    mixture = MixtureOfLogistics(input_size=param["input_size"], nb_centers = param["nb_component"], transform_mean=param["transform_mean"], transform_std=param["transform_std"])
    path_weights = os.path.join(param["model_dir"], "mixture_of_logistics.pt")
    mixture.load_state_dict(torch.load(path_weights))
    mixture_imputation = MixtureOfLogisticsImputation(mixture)
    return mixture_imputation

