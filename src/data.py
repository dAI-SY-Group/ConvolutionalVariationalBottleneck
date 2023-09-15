import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.transforms import *
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

from src.utils import get_available_datasets, stat_to_tensor

def load_victim_data(victim_data_path, batch_size):
    victim_data_path = victim_data_path if victim_data_path.endswith('.tdump') else victim_data_path
    if os.path.exists(victim_data_path):
        vic_data = torch.load(victim_data_path)
        print(f'Loaded victim data from {victim_data_path}!')
        data_shape = vic_data['inputs'][0].shape 
        return list(vic_data['inputs'].split(batch_size)), list(vic_data['targets'].split(batch_size)), data_shape
    else:
        raise ValueError(f'Tried loading victim data from {victim_data_path}, but that does not exist. Try another path and make sure this victim dataset was already generated!')


class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for handling input data and labels.

    Args:
        data (array-like): Input data.
        targets (array-like): Labels corresponding to the input data.
        transforms (callable, optional): A function/transform to apply to the input data.
        target_transforms (callable, optional): A function/transform to apply to the labels.
        manual_transform (bool, optional): Whether to manually convert data and targets to tensors.

    Attributes:
        data (torch.Tensor or array-like): Input data.
        targets (torch.Tensor or array-like): Labels corresponding to the input data.
        transforms (callable, optional): A function/transform to apply to the input data.
        target_transforms (callable, optional): A function/transform to apply to the labels.
        target_distribution (dict): Distribution of class labels in the dataset.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves the item at the specified index.
        __str__(): Returns a string representation of the dataset.
        __repr__(): Returns a detailed string representation of the dataset.

    """
    def __init__(self, data, targets, transforms=None, target_transforms=None, manual_transform=False):
        """
        Initialize Dataset with input data, labels, and optional transformations.

        If `manual_transform` is True, data and targets are converted to tensors manually.

        Args:
            data (array-like): Input data.
            targets (array-like): Labels corresponding to the input data.
            transforms (callable, optional): A function/transform to apply to the input data.
            target_transforms (callable, optional): A function/transform to apply to the labels.
            manual_transform (bool, optional): Whether to manually convert data and targets to tensors.

        """
        if manual_transform:
            self.data = torch.FloatTensor(np.array(data))
            self.targets = torch.LongTensor(targets)
        else:
            self.data = np.array(data)
            self.targets = np.array(targets)
        self.transforms = transforms
        self.target_transforms = target_transforms
        #calculate and store number of class labels of the dataset
        self.target_distribution = defaultdict(int, dict(pd.Series(self.targets).value_counts().sort_index()))

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.

        """
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Retrieves the item at the specified index. If transforms are specified, they are applied to the data and labels.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the input data and its corresponding label.

        """
        x = self.data[idx]
        y = self.targets[idx]
        if self.transforms:
            x = self.transforms(x)
        if self.target_transforms:
            y = self.target_transforms(y)
        return x, y
    
    def __str__(self):
        """
        Returns a string representation of the dataset.

        Returns:
            str: String representation of the dataset.

        """
        return f'Dataset size: {len(self)}. Input shape: {self.data.shape}. Output shape: {self.targets.shape}.\nTransforms: {self.transforms}.\nTarget Transforms: {self.target_transforms}.\nTarget Distribution: {self.target_distribution}.'
    
    def __repr__(self):
        """
        Returns a detailed string representation of the dataset.

        Returns:
            str: Detailed string representation of the dataset.

        """
        return self.__str__()
    
class ClientDataset(Dataset):
    """
    Custom dataset class for a specific client in a federated learning scenario. Inherits from SimpleAI.data.datasets.bases.Dataset

    Args:
        id (int): Identifier for the client.
        data (array-like): Input data.
        targets (array-like): Labels corresponding to the input data.
        transforms (callable, optional): A function/transform to apply to the input data.
        target_transforms (callable, optional): A function/transform to apply to the labels.

    Attributes:
        id (int): Identifier for the client.
        Inherits all attributes from the parent class Dataset.

    Methods:
        __init__(id, data, targets, transforms=None, target_transforms=None): Initializes ClientDataset.
        __str__(): Returns a string representation of the client dataset.

    """
    def __init__(self, id, data, targets, transforms=None, target_transforms=None):
        """
        Initialize ClientDataset for a specific client.

        Args:
            id (int): Identifier for the client.
            data (array-like): Input data.
            targets (array-like): Labels corresponding to the input data.
            transforms (callable, optional): A function/transform to apply to the input data.
            target_transforms (callable, optional): A function/transform to apply to the labels.

        """
        super().__init__(data, targets, transforms, target_transforms)
        self.id = id
    
    def __str__(self):
        """
        Returns a string representation of the client dataset.

        Returns:
            str: String representation of the client dataset.

        """
        return f'ClientDataset for Client {self.id}\n' + super().__str__()

class FederatedDataset(TorchDataset):
    def __init__(self, dataset, client_list, distribution_config='IID', preset_label_distribution=None, transforms = torchvision.transforms.ToTensor(), seed=42):
        assert distribution_config is not None, 'distribution_config must not be None for FederatedDatasets!'
        self.dataset = dataset
        self.targets = [sample[1] for sample in self.dataset]
        self.client_list = client_list
        self.distribution_config = distribution_config
        self.preset_label_distribution = preset_label_distribution
        self.transforms = transforms
        self.seed = seed
        
        #self.class_distribution = class_distribution
        self.idx_distribution = self.get_index_distribution(preset_label_distribution)

        self.client_datasets = {client: None for client in self.client_list}

        for client in self.client_list:
            self.get_client_dataset(client, True)

    def __len__(self):
        return len(self.client_list)

    def __str__(self):
        return f'FederatedDataset:\n  Partition Mode: {self.distribution_config}\n  Clients: {self.client_list}\n  Samples per Client: {[len(ds) for ds in self.client_datasets.values()]}'

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, index):
        assert index in self.client_list or index in list(range(len(self.client_list))), f'The index {index} is not part of the client_list ({self.client_list})!'
        if index not in self.client_list:
            index = self.client_list[index]
        if self.client_datasets[index] is None:
            self.get_client_dataset(index, True)
        return self.client_datasets[index]

    def get_client_dataset(self, index, reload=False):
        assert index in self.client_list or index in list(range(len(self.client_list))), f'The index {index} is not part of the client_list ({self.client_list})!'
        if reload or self.client_datasets[index] is None:
            client_data = []
            client_targets = []
            for sample_index in self.idx_distribution[index]:
                client_data.append(np.array(self.dataset[sample_index][0]))
                client_targets.append(self.dataset[sample_index][1])
            self.client_datasets[index] = ClientDataset(index, client_data, client_targets, self.transforms, None)
        return self.client_datasets[index]
        
    #distributes data as in: https://fedlab.readthedocs.io/en/master/tutorials/cifar10_tutorial.html#data-cifar10. --> See descriptions
    def get_index_distribution(self, preset_global_distribution=None):
        """
        Calculates the index distribution (i.e. how samples of the whole central base dataset are distributed between clients) for clients based on distribution configuration or preset.

        Args:
            preset_global_distribution (dict, optional): Predefined global label distribution.

        Returns:
            dict: Index distribution for clients.

        """
        distribution = {}

        num_clients = len(self.client_list)
        N = len(self.dataset)

        if preset_global_distribution is not None:
            #based on a given global preset distribution create a new index distribution for the targets that follows the same class distribution
            def create_index_distribution(targets, global_preset_distribution):
                #from the global_distribution create a matric where each row is a client and each column is a class, the value is the corresponding nnumber of samples each client has for that class
                num_clients = len(global_preset_distribution)
                num_classes = len(np.unique(targets))
                client_distribution = np.zeros((num_clients, num_classes))+1e-10
                for client, client_distribution_ in global_preset_distribution.items():
                    for class_, num_samples in client_distribution_.items():
                        client_distribution[client, class_] = num_samples
                #transform the total number of samples for each class to a distribution
                p_client_distribution = client_distribution / np.sum(client_distribution, axis=0, keepdims=True)
                #for each class get the indices of the target list
                class_indices = {class_: np.where(targets == class_)[0] for class_ in np.unique(targets)}
                class_avaliable_samples = {class_: len(class_indices[class_]) for class_ in class_indices}
                #shuffle the indices for each class
                for class_ in class_indices:
                    np.random.shuffle(class_indices[class_])

                #create a new index distribution that follows the same distribution as the global preset distribution
                target_index_distribution = {}
                for client_index, client_distribution_ in enumerate(p_client_distribution):
                    target_index_distribution[client_index] = []
                    for class_index, class_percentage in enumerate(client_distribution_):
                        num_given_samples = int(class_percentage*class_avaliable_samples[class_index])
                        target_index_distribution[client_index] += list(class_indices[class_index][:num_given_samples])
                        class_indices[class_index] = class_indices[class_index][num_given_samples:]    
                for client_index in range(num_clients):
                    target_index_distribution[client_index] = np.array(target_index_distribution[client_index])
                return target_index_distribution
            distribution = create_index_distribution(self.targets, preset_global_distribution)
        else:
            #batch_idx defines which client gets which samples by idx
            if self.distribution_config in ['homo', 'homogenous', 'iid', 'IID']:
                idx = np.random.permutation(N)
                batch_idx = np.array_split(idx, num_clients)
            else:
                raise NotImplementedError(f'Partition mode {self.distribution_config} is not implemented yet!')
            distribution = {client: batch_idx[i] for i, client in enumerate(self.client_list)}
        return distribution




def get_dataloaders(dataset, is_federated, config):
    """
    Get the dataloaders for the given dataset

    Args:
        dataset (str): name of the dataset
        is_federated (bool): whether to return a federated dataloader or a central dataloader
        config (Munch or str): either a Munch object containing the config for the dataset or the name of the dataset to load a standard config for

    Returns:
        tuple: either (trn_loader, tst_loader, val_loader) or (fed_data)
        
    """
    available_datasets = get_available_datasets()
    assert dataset in available_datasets.keys(), f'Dataset {dataset} is not available! Available datasets are {available_datasets.keys()}'

    trn_transformations = get_transforms(config.data.train_transformations, (stat_to_tensor(config.data.mean), stat_to_tensor(config.data.std))) if config.data.train_transformations is not None else None
    val_transformations = get_transforms(config.data.val_transformations, (stat_to_tensor(config.data.mean), stat_to_tensor(config.data.std))) if config.data.val_transformations is not None else None
    print(f'Using the following transformations for training data: {trn_transformations}')
    print(f'Using the following transformations for validation and test data: {val_transformations}')

    if is_federated:
        dataloaders = get_federated_dataloader(config, trn_transformations, val_transformations) # returns "fed_trn_loader, fed_tst_loader, fed_val_loader"
    else:
        dataloaders = get_central_dataloader(config, trn_transformations, val_transformations) # returns "trn_loader, tst_loader, val_loader"

    if 'RandomCrop' in config.data.train_transformations.keys():
        config.data.shape[1] = config.data.train_transformations['RandomCrop'][0][0]
        config.data.shape[2] = config.data.train_transformations['RandomCrop'][0][1]
    if 'Resize' in config.data.train_transformations.keys():
        config.data.shape[1] = config.data.train_transformations['Resize'][0][0]
        config.data.shape[2] = config.data.train_transformations['Resize'][0][1]
    print(f'Final data shape: {config.data.shape}')
    return dataloaders

def get_federated_dataloader(config, trn_transformations, val_transformations):    
    """
    Generates federated data loaders based on the provided configuration.

    Args:
        config (Munch): The configuration object containing various settings.
        trn_transformations (callable): The transformations to apply to the training data.
        val_transformations (callable): The transformations to apply to the validation data.

    Returns:
        tuple: A tuple containing three dictionaries of DataLoader objects
               (`fed_trn_loader`, `fed_tst_loader`, `fed_val_loader`).

    Example:
        >>> config = build_config('...')
        >>> trn_transforms = ...
        >>> val_transforms = ...
        >>> trn_loader, tst_loader, val_loader = get_federated_dataloader(config, trn_transforms, val_transforms)

    """
    distribution_config = config.data_distribution_config

    #create all the actual federated data loaders
    fed_trn_loader = {}
    fed_tst_loader = {}
    fed_val_loader = {}

    client_list = list(range(config.training.num_clients))        
    #load the central dataset as a base
    tmp_config = config.copy()
    tmp_config.data.create_validation_split = 0
    trn_loader, tst_loader, val_loader = get_central_dataloader(tmp_config, get_transforms({'force_none':True}), get_transforms({'force_none':True}))

    print(f'Splitting central dataset into federated client datasets with a IID partitioning!')
    #create federated datasets from it
    fed_trn_set = FederatedDataset(trn_loader.dataset, client_list, 'IID', None, trn_transformations, config.seed) if trn_loader is not None else {client: None for client in client_list}
    global_preset_distribution = {client: client_dataset.target_distribution for client, client_dataset in fed_trn_set.client_datasets.items()}
    fed_tst_set = FederatedDataset(tst_loader.dataset, client_list, 'IID', global_preset_distribution, val_transformations, config.seed) if tst_loader is not None else {client: None for client in client_list}
    fed_val_set = FederatedDataset(val_loader.dataset, client_list, 'IID', global_preset_distribution, val_transformations, config.seed) if val_loader is not None else {client: None for client in client_list}

    print(f'Federated DataLoaders are built according to the following federated Datasets:\n  Train {fed_trn_set}\n  Test {fed_tst_set}\n  Validation {fed_val_set}')

    for client in client_list:
        trn_set = fed_trn_set[client]
        tst_set = fed_tst_set[client]
        val_set = fed_val_set[client]

        if val_set is None and config.data.create_validation_split:
            print(f'Creating validation split of {config.data.create_validation_split} from training set for client {client}')
            trn_set, val_set = split_dataset(trn_set, (1-config.data.create_validation_split, config.data.create_validation_split), shuffle = True, seed=config.seed, ds1_transforms=trn_transformations, ds2_transforms=val_transformations, dataset_class=ClientDataset, id=client)

        trn_loader = DataLoader(trn_set, batch_size=min(config.training.batch_size, len(trn_set)), shuffle=config.data.shuffle, drop_last=False) if trn_set is not None else None
        tst_loader = DataLoader(tst_set, batch_size=min(config.training.batch_size, len(tst_set)), shuffle=False, drop_last=False) if tst_set is not None else None
        val_loader = DataLoader(val_set, batch_size=min(config.training.batch_size, len(val_set)), shuffle=False, drop_last=False) if val_set is not None else None  

        fed_trn_loader[client] = trn_loader
        fed_tst_loader[client] = tst_loader
        fed_val_loader[client] = val_loader

        print(f'Client {client} | Trainbatches: {len(trn_loader) if trn_loader is not None else 0} | Testbatches: {len(tst_loader) if tst_loader is not None else 0} | Validationbatches: {len(val_loader) if val_loader is not None else 0}')
                        
    return fed_trn_loader, fed_tst_loader, fed_val_loader



def get_central_dataloader(config, trn_transformations, val_transformations):
    """
    Returns central data loaders for training, testing, and validation sets.

    This function prepares data loaders for the specified dataset using the provided configurations
    and transformations.

    Parameters:
        config (Munch): Configuration object containing dataset and training settings.
        trn_transformations (callable): Transformation function for training set.
        val_transformations (callable): Transformation function for validation set.

    Returns:
        tuple: A tuple containing three data loaders (training, testing, and validation).
               If a specific set is not available, the corresponding loader will be None.

    Raises:
        (various): This function may raise various exceptions depending on the specific dataset
                   loading and transformation functions used.

    Example:
        >>> config = build_config('...')
        >>> trn_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        >>> val_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        >>> trn_loader, tst_loader, val_loader = get_central_dataloader(config, trn_transforms, val_transforms)

    Note:
        If the validation set is not provided but `create_validation_split` is enabled in the
        configuration, a split will be created from the training set.
    """
    get_dataset = eval(f'get_{config.data.dataloader}_dataset')
    trn_set, tst_set, val_set = get_dataset(config, trn_transformations, val_transformations)

    if val_set is None and config.data.create_validation_split:
        print(f'Creating validation split of {config.data.create_validation_split} from training set')
        trn_set, val_set = split_dataset(trn_set, (1-config.data.create_validation_split, config.data.create_validation_split), shuffle = True, ds1_transforms=trn_transformations, ds2_transforms=val_transformations)

    trn_loader = DataLoader(trn_set, batch_size=min(config.training.batch_size, len(trn_set)), shuffle=True, drop_last=True) if trn_set is not None else None
    tst_loader = DataLoader(tst_set, batch_size=min(config.training.batch_size, len(tst_set)), shuffle=False, drop_last=False) if tst_set is not None else None
    val_loader = DataLoader(val_set, batch_size=min(config.training.batch_size, len(val_set)), shuffle=False, drop_last=False) if val_set is not None else None

    print(f'Created central dataloaders for {config.data.dataset} dataset. Transformed them with:')
    print(str(trn_transformations))

    print(f'Batchsize: {config.training.batch_size} | Trainbatches: {len(trn_loader) if trn_loader is not None else 0} | Testbatches: {len(tst_loader) if tst_loader is not None else 0} | Validationbatches: {len(val_loader) if val_loader is not None else 0}')
    return trn_loader, tst_loader, val_loader


def get_image_dataset(config, trn_transformations, val_transformations):
    """
    Get image dataset.

    Args:
        config: Configuration (Munch) object.
        trn_transformations: Training transformations.
        val_transformations: Validation transformations.

    Returns:
        Dataset: Training dataset.
        Dataset: Testing dataset.
        Dataset: Validation dataset.

    """
    datapath = os.path.expanduser(config.dataset_path)

    if config.data.dataset == 'CIFAR10':
        trn_set, tst_set = _build_cifar10(datapath, trn_transformations, val_transformations)
        val_set = None
    elif config.data.dataset == 'MNIST':
        trn_set, tst_set = _build_mnist(datapath, trn_transformations, val_transformations)
        val_set = None
    elif config.data.dataset in ['MedMNISTDerma', 'MedMNISTPneumonia', 'MedMNISTRetina', 'MedMNISTBlood']:
        datapath = os.path.join(datapath, 'MedMNIST', f'{config.data.dataset[8:]}', f'{config.data.dataset[8:].lower()}mnist.npz')
        trn_set, tst_set, val_set = _build_med_mnist(datapath, trn_transformations, val_transformations)
    else:
        ValueError(f'Dataloaders for the {config.data.dataset} dataset are not yet implemented!')
    return trn_set, tst_set, val_set

def _build_cifar10(datapath, train_transformations = transforms.ToTensor(), val_transformations = transforms.ToTensor()):
    """
    Builds CIFAR-10 datasets for training and testing.

    Args:
        datapath (str): Path to store the CIFAR-10 dataset.
        train_transformations (torchvision.transforms.Compose, optional): Transformations for training set. Default is transforms.ToTensor().
        val_transformations (torchvision.transforms.Compose, optional): Transformations for validation set. Default is transforms.ToTensor().

    Returns:
        Dataset: Training set.
        Dataset: Testing set.

    """
    trn_set = torchvision.datasets.CIFAR10(root=datapath, train=True, download=True)
    tst_set = torchvision.datasets.CIFAR10(root=datapath, train=False, download=True)
    trn_set = Dataset(trn_set.data, trn_set.targets, train_transformations)
    tst_set = Dataset(tst_set.data, tst_set.targets, val_transformations)
    return trn_set, tst_set

def _build_mnist(datapath, train_transformations = transforms.ToTensor(), val_transformations = transforms.ToTensor()):
    """
    Builds MNIST datasets for training and testing.

    Args:
        datapath (str): Path to store the MNIST dataset.
        train_transformations (torchvision.transforms.Compose, optional): Transformations for training set. Default is transforms.ToTensor().
        val_transformations (torchvision.transforms.Compose, optional): Transformations for validation set. Default is transforms.ToTensor().

    Returns:
        Dataset: Training set.
        Dataset: Testing set.

    """
    trn_set = torchvision.datasets.MNIST(root=datapath, train=True, download=True)
    tst_set = torchvision.datasets.MNIST(root=datapath, train=False, download=True)
    trn_set = Dataset(trn_set.data, trn_set.targets, train_transformations)
    tst_set = Dataset(tst_set.data, tst_set.targets, val_transformations)
    return trn_set, tst_set


def _build_med_mnist(datapath, train_transformations = transforms.ToTensor(), val_transformations = transforms.ToTensor()):
    """
    Builds a MedMNIST dataset for training, testing, and validation.

    Args:
        datapath (str): Path to the MedMNIST dataset.
        train_transformations (torchvision.transforms.Compose, optional): Transformations for training set. Default is transforms.ToTensor().
        val_transformations (torchvision.transforms.Compose, optional): Transformations for validation and test sets. Default is transforms.ToTensor().

    Returns:
        Dataset: Training set.
        Dataset: Testing set.
        Dataset: Validation set.

    """
    class MedMNIST(Dataset):
        """
        MedMNIST dataset class. Dataset from https://medmnist.com/

        Args:
            datapath (str): Path to the MedMNIST dataset.
            split (str): Specifies the dataset split (e.g., 'train', 'val', 'test').
            transform (callable, optional): A function/transform to apply to the data. Default is transforms.ToTensor().

        Attributes:
            datapath (str): Path to the MedMNIST dataset.
            split (str): Specifies the dataset split (e.g., 'train', 'val', 'test').
            transforms (callable, optional): A function/transform to apply to the data.
            data (numpy.ndarray): Array of images.
            targets (numpy.ndarray): Array of labels.

        Methods:
            get_np(index): Returns the numpy representation of an image at the specified index.
            __getitem__(index): Gets an item (image and label) from the dataset at the specified index.
            __len__(): Returns the total number of samples in the dataset.

        """
        def __init__(self, datapath, split, transforms=transforms.ToTensor()):
            self.datapath = datapath
            self.split = split
            self.transforms = transforms
            npz_data = np.load(datapath)
            self.data = npz_data[f'{split}_images']
            self.targets = npz_data[f'{split}_labels']
            assert len(self.data) == len(self.targets)

        def get_np(self, index):
            """
            Returns the numpy representation of an image at the specified index.

            Args:
                index (int): Index of the image.

            Returns:
                numpy.ndarray: Numpy array representing the image.

            """
            return self.data[index]

        def __getitem__(self, index):
            """
            Gets an item (image and label) from the dataset at the specified index.

            Args:
                index (int): Index of the item.

            Returns:
                tuple: Tuple containing the transformed image and its label.

            """
            sample = self.data[index]
            if self.transforms is not None:
                sample = self.transforms(sample)
            return sample, self.targets[index][0]

        def __len__(self):
            return len(self.targets)

    return MedMNIST(datapath, 'train', train_transformations), MedMNIST(datapath, 'test', val_transformations), MedMNIST(datapath, 'val', val_transformations)




def split_dataset(dataset, split=(0.9, 0.1), shuffle=True, seed=42, ds1_transforms=None, ds2_transforms=None, ds1_target_transforms=None, ds2_target_transforms=None, dataset_class=Dataset, *args, **kwargs):
    """
    Split a dataset into two datasets.

    Args:
        dataset (Dataset): The original dataset.
        split (tuple): A tuple specifying the split ratio.
        shuffle (bool): Whether to shuffle the dataset before splitting.
        seed (int): Seed for randomization.
        ds1_transforms (list, 'same', optional): Transforms for the first dataset.
        ds2_transforms (list, 'same', optional): Transforms for the second dataset.
        ds1_target_transforms (list, 'same', optional): Target transforms for the first dataset.
        ds2_target_transforms (list, 'same', optional): Target transforms for the second dataset.
        dataset_class (type, optional): Class type of the dataset.

    Returns:
        Dataset: The first split of the dataset.
        Dataset: The second split of the dataset.
    """
    assert sum(split) == 1, 'split must sum to 1'
    if shuffle:
        torch.manual_seed(seed)
        indices = torch.randperm(len(dataset)).tolist()
    else:
        indices = list(range(len(dataset)))
    split1 = int(split[0] * len(dataset))
    if ds1_transforms == 'same':
        ds1_transforms = dataset.transforms
    if ds2_transforms == 'same':
        ds2_transforms = dataset.transforms
    if ds1_target_transforms == 'same':
        ds1_target_transforms = dataset.target_transforms
    if ds2_target_transforms == 'same':
        ds2_target_transforms = dataset.target_transforms
    ds1 = dataset_class(data = dataset.data[indices[:split1]], targets = dataset.targets[indices[:split1]], transforms=ds1_transforms, target_transforms=ds1_target_transforms, *args, **kwargs)
    ds2 = dataset_class(data = dataset.data[indices[split1:]], targets = dataset.targets[indices[split1:]], transforms=ds2_transforms, target_transforms=ds2_target_transforms, *args, **kwargs)
    return ds1, ds2


def get_transforms(transforms_dict={}, norm_parameters=None):
    """
    Generate a composition of PyTorch image transformations.

    Args:
        transforms_dict (dict): A dictionary where keys are the names of torchvision.transforms transformations
            (e.g., 'Resize', 'Normalize') and values are tuples of parameters for the transformation.
        norm_parameters (tuple or None): Parameters for normalization. Used when 'Normalize' transform is specified.

    Returns:
        torchvision.transforms.Compose or None: A composition of transformations, or None if no transforms are specified.
    """

    trans_list = []
    if 'force_none' in transforms_dict:
        trans_list = []
    elif len(transforms_dict.keys()) > 0:
        for transform, parameters in transforms_dict.items():
            #print(transform, parameters)
            if transform == 'Normalize':
                if len(parameters) == 0:
                    parameters = norm_parameters
                assert parameters is not None, f'Normalization parameters are not specified!'
            t = eval(f'{transform}(*parameters)')
            if transform == 'Grayscale': # make sure to grayscale first
                trans_list.insert(0,t)
            else:
                trans_list.append(t)
    else:
        trans_list.append(transforms.ToTensor())
    return transforms.Compose(trans_list) if len(trans_list) > 0 else None



def Grayscale_to_RGB(*args, **kwargs):
    """
    Convert grayscale images to RGB format.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        torchvision.transforms.Lambda: A lambda function that repeats input channels to convert grayscale to RGB.
    """
    return transforms.Lambda(lambda x: x.repeat([3, 1, 1], 0))