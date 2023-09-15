import os
from copy import deepcopy

import torch

from src.models.CNN import build_CNN
from src.models.ViT import build_VisionTransformer
from src.models.ResNet import build_ResNet

def get_model(architecture, config):

    if 'CNN' in architecture:
        model = build_CNN(architecture, config)
    elif 'ResNet' in architecture:
        model = build_ResNet(architecture, config)
    elif 'VisionTransformer' in architecture:
        model = build_VisionTransformer(architecture, config)
    else:
        raise NotImplementedError(
            f'The model architecture {architecture} is not implemented yet..')
    print(f'Loaded model with {architecture} architecture, input shape {config.data.shape}, {config.num_classes} classes.')
    
    model = attach_model_functionalities(model, config)

    return model

def attach_model_functionalities(model, config, verbose=True):
    if config.checkpoint_path:
        model.savepath = os.path.join(config.checkpoint_path, config.experiment_name)
        model.save = lambda path=model.savepath: save_model(path, model)
        model.load = lambda path=model.savepath: load_model(path, model)
        print(f'Setting the models checkpoint path to {model.savepath}.ckpt')

    model.reset_parameters = lambda verbose=False: reset_parameters(model, verbose)

    model.copy = lambda: copy_model(model)

    model._config = config

    return model



def save_model(path, model):
    """
    Save the model to a file.

    Args:
        path (str): The file path to save the model.
        model (torch.nn.Module): The neural network model to be saved.

    Returns:
        None

    Example:
        >>> save_model('my_model.ckpt', MyModel())

    """
    path = path if path.endswith('.ckpt') or path.endswith('.state') else path + '.ckpt'
    torch.save(model.state_dict(), path)


def load_model(path, model):
    """
    Load a model from a file.

    Args:
        path (str): The file path to load the model from.
        model (torch.nn.Module): The neural network model.

    Returns:
        torch.nn.Module: The loaded neural network model.

    Example:
        >>> loaded_model = load_model('my_model.ckpt', MyModel())

    Raises:
        FileNotFoundError: If the specified checkpoint file is not found.

    """
    path = path if path.endswith('.ckpt') or path.endswith('.state') else path + '.ckpt'
    if os.path.isfile(path):
        load_dict = torch.load(path)
        model.load_state_dict(load_dict)
        print(f'Loaded model from {path}...')
        return model
    else:
        print(f'Checkpoint file {path} was not found. No other weights were loaded into the model...')
        return model

def reset_parameters(model, verbose=False):
    """
    Recursively resets the parameters of a PyTorch model and its sub-modules.

    Args:
        model (nn.Module): The PyTorch model to reset parameters for.
        verbose (bool, optional): Whether to print verbose messages. Defaults to False.
    """
    children = list(model.children())
    for child in children:
        if len(list(child.children())) > 0:
            reset_parameters(child, verbose)
        else:
            try:
                child.reset_parameters()
                if verbose:
                    print(f'Resetting parameters of {child}!')
            except:
                if verbose:
                    print(f'{child} has no parameters to be reset!')
                continue

def copy_model(model):
    config = model._config
    model_copy = deepcopy(model)
    model_copy = attach_model_functionalities(model_copy, config, verbose=False)
    return model_copy