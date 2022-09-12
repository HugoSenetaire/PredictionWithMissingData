import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from .utils import log_prob_from_logits
import os
from .mixture_of_logistics import MixtureOfLogistics
import sklearn
import numpy as np
import matplotlib.pyplot as plt

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


def initialize_mixture(mixture, train_dataset, max_size = 100000, epochs = 2, lr = 1e-4, batch_size = 64 , validation_dataset = None, dir = None):
    """
    Initialize the mixture of logistics with a k-means algorithm. Centers and weights are initialized directly with kmeans.
    We use a simple SGD to optimize the sigma parameters.
    To evaluate kmeans we use a dataset of size max_size.

    Parameters
    ----------
    mixture : MixtureOfLogistics
        Mixture of logistics to initialize.

    loader : torch.utils.data.DataLoader
        DataLoader of the dataset to use to initialize the mixture.
    
    max_size : int
        Maximum size of the dataset to use to initialize the mixture.

    epochs : int
        Number of epochs to use to optimize the sigma parameters.

    lr : float
        Learning rate to use to optimize the sigma parameters.

    batch_size : int
        Batch size to use to optimize the sigma parameters.
    """

    index = np.random.choice(range(len(train_dataset)), min(max_size,len(train_dataset)), replace=False)
    try :
        current_train_dataset = train_dataset.data[index]
    except :
        current_train_dataset = [train_dataset.__getitem__(i) for i in index]
        current_train_dataset = torch.cat(current_train_dataset, dim = 0)
    
    current_train_dataset = mixture.transform_data(current_train_dataset)
    data_shape = current_train_dataset.shape
    nb_centers = mixture.nb_centers


    kmeans = sklearn.cluster.KMeans(n_clusters = nb_centers, n_init = 1).fit(current_train_dataset.flatten(1).detach().cpu().numpy())
    mixture.mu_parameters.data = torch.from_numpy(kmeans.cluster_centers_).reshape((nb_centers, *data_shape[1:])).to(mixture.mu_parameters.device)
    repartition = kmeans.labels_
    weights = torch.nn.functional.one_hot(torch.from_numpy(repartition).long(), nb_centers).to(device = mixture.log_weight.device, dtype = torch.float32)
    mixture.log_weight.data = log_prob_from_logits(torch.log(weights.float().mean(dim=0) +1e-8))

    print("Repartition after kmeans : {}".format(weights.mean(dim=0)))
    print("Weights after kmeans : {}".format(torch.exp(mixture.log_weight).detach().cpu().numpy()))
    # print("mu after kmeans : {}".format(mixture.mu_parameters.detach().cpu().numpy()))

    for k in range(nb_centers) :
        current_train_dataset_k = current_train_dataset[repartition == k].flatten(1)
        aux_sigma = current_train_dataset_k.std(dim=0).detach().cpu().numpy()
        # print("Sigma for center {} : {}".format(k, aux_sigma))
        mixture.log_s_parameters.data[k] = torch.log(current_train_dataset_k.std(dim = 0).reshape(data_shape[1:]).to(mixture.log_s_parameters.device) + 1e-8)

    # print(mixture.log_s_parameters.data)
    # mixture.log_weight.requires_grad = False
    # mixture.mu_parameters.requires_grad = False
    # Training the sigma parameters of the mixture
    # mixture.log_s_parameters.requires_grad = True
    # sgd_training(mixture = mixture, train_dataset = train_dataset, validation_dataset= validation_dataset, batch_size = batch_size, epochs = epochs, lr = lr)
    # print(mixture.log_s_parameters[0]) 
    # mixture.log_weight.requires_grad = True
    # mixture.mu_parameters.requires_grad = True

def validation(mixture, k, epochs, dataloader, verbose = True, sample = True, dir = None):
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
    
    if sample and dir is not None:
        img = mixture.sample(None, None, nb_samples = 10)
        img = img.reshape((10, *data.shape[1:])).permute(0,2,3,1).detach().cpu().numpy()
        sample_dir = os.path.join(os.path.abspath(dir), "epoch_{}".format(k))       
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        
        mixture = mixture.mu_parameters.data * 127.5 + 127.5
        mixture = mixture[:10].reshape((10, *data.shape[1:])).permute(0,2,3,1) 

        mixture = mixture.cpu().numpy().astype(np.uint8)
        for center in range(10):
            plt.figure(figsize=(10,10))
            out_path = os.path.join(sample_dir, "center_{}.png".format(center))
            plt.imshow(mixture[center], cmap='gray')
            plt.savefig(out_path)
            plt.close()
        for i in range(10):
            plt.figure(figsize=(20,10))
            plt.imshow(img[i], cmap = 'gray')
            out_path = os.path.join(sample_dir, "sample_{}.png".format(i))
            plt.savefig(out_path)
            plt.close()
        for i in range(10):
            plt.figure(figsize=(20,10))
            plt.imshow(data[i].permute(1,2,0).detach().cpu().numpy(), cmap = 'gray')
            out_path = os.path.join(sample_dir, "real_{}.png".format(i))
            plt.savefig(out_path)
            plt.close()

def sgd_training(epochs, mixture, train_dataset, validation_dataset, batch_size, lr=1e-4, verbose = True, weights = None, dir = None):
    optimizer = torch.optim.Adam(mixture.parameters(), lr=lr)

    # build dataloaders on top of datasets
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    if validation_dataset is not None :
        val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        validation(mixture, -1, epochs, val_dataloader, dir = dir)

    # Training loop
    for k in range(epochs):
        print("=========================================")
        print("Epoch {}".format(k))
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            data = data.to(next(mixture.parameters()).device)
            log_prob = mixture.forward(data, dependency = weights)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mixture.mu_parameters.data = torch.clamp(mixture.mu_parameters.data, -1.0, 1.0)

            if verbose and i % max((len(dataloader)//10),1) == 0:
                print("Batch {} / {}".format(i, len(dataloader)))
                print("Loss : {}".format(loss.item()))
        
        # Validation
        if validation_dataset is not None :
            validation(mixture, k, epochs, val_dataloader, dir = dir)

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
        # dependency_index = torch.argmax(dependency, dim = 0)
        # dependency = torch.nn.functional.one_hot(dependency_index, num_classes = mixture.nb_centers).transpose(0,1)

        log_prob = mixture.forward(data, mask = None, dependency = dependency)
        loss = -log_prob.mean()
        loss.backward()
        optimizer.step()
        mixture.mu_parameters.data = torch.clamp(mixture.mu_parameters.data, -1.0, 1.0)

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
    

def em_training(mixture, train_dataset, validation_dataset, nb_m_step = 1, nb_e_step = None, nb_loop = 20, verbose = True, lr = 1e-4, batch_size = 64, dir = None):
    dataloader = iter(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0))
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mixture.log_weight.requires_grad = False
    optimizer = torch.optim.Adam(mixture.parameters(), lr=lr)
    validation(mixture, -1, nb_loop, val_dataloader, verbose = verbose, dir = dir)
    for k in range(nb_loop):
        print("=========================================")
        print("Loop {}".format(k))
        e_step(mixture = mixture, dataloader = dataloader, nb_e_step = nb_e_step)
        m_step(mixture = mixture, dataloader = dataloader, nb_m_step = nb_m_step, optimizer= optimizer)
        validation(mixture, k, nb_loop, val_dataloader, verbose = verbose, dir = dir)
        


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
                            verbose = True,
                            max_size_init = 100000,):
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
    initialize_mixture(mixture, train_dataset, max_size = max_size_init, epochs = 2, lr = lr, batch_size = batch_size, validation_dataset=validation_dataset, dir = model_dir)
    if use_cuda :
        mixture = mixture.to("cuda:0")

    if type_of_training == "sgd":
        sgd_training(epochs = epochs,
                    mixture = mixture,
                    train_dataset = train_dataset,
                    validation_dataset = validation_dataset,
                    lr = lr,
                    batch_size = batch_size,
                    verbose = verbose,
                    dir = model_dir)

    elif type_of_training == "em":
        em_training(mixture = mixture,
                    train_dataset = train_dataset,
                    validation_dataset = validation_dataset,
                    batch_size = batch_size,
                    lr=lr,
                    nb_e_step=nb_e_step,
                    nb_m_step=nb_m_step,
                    nb_loop=epochs,
                    verbose = verbose,
                    dir = model_dir)
    else :
        raise ValueError("type_of_training should be either sgd or em")
    
    # Save the model
    torch.save(mixture.state_dict(), os.path.join(model_dir, f"mixture_of_logistics.pt"))
    

