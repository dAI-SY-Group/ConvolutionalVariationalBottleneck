from munch import Munch

import numpy as np
import torch
import torch.nn as nn

from src.models.modules import MLPBlock, prepare_VB_config, VBModule, get_layer

def build_CNN(architecture, config):    
    conv_activation_function = config.model.conv_activation_function
    conv_regularization_layer = config.model.regularization_layer
    mlp_activation_function = config.model.mlp_activation_function
    mlp_regularization_layer = config.model.mlp_regularization_layer
    pooling_layer = config.model.pooling_layer
    if architecture == 'CNN':
        model = CNN(config.data.shape, config.num_classes, channels=config.model.channels, kernels=config.model.kernels, stride=config.model.stride, padding=config.model.padding, widths=config.model.widths, pooling_layer=pooling_layer, pooling_positions=config.model.pooling_positions, conv_regularization_layer=conv_regularization_layer, conv_activation_function=conv_activation_function, mlp_regularization_layer=mlp_regularization_layer, mlp_activation_function=mlp_activation_function, headless=config.model.headless)
    elif architecture =='VBCNN':
        model = VBCNN(config.data.shape, config.num_classes, channels=config.model.channels, kernels=config.model.kernels, stride=config.model.stride, padding=config.model.padding, widths=config.model.widths, pooling_layer=pooling_layer, pooling_positions=config.model.pooling_positions, conv_regularization_layer=conv_regularization_layer, conv_activation_function=conv_activation_function, mlp_regularization_layer=mlp_regularization_layer, mlp_activation_function=mlp_activation_function, headless=config.model.headless, VB_config=prepare_VB_config(config.model.VB))
    else:
        raise ValueError(f'The model architecture {architecture} is not implemented yet..')
    return model

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros', regularization_layer=None, activation_function=None, pooling_layer=None, flatten=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.regularization_layer = get_layer(regularization_layer.name, regularization_layer.parameters, channels = out_channels) if regularization_layer is not None else None
        self.activation_function = get_layer(activation_function.name, activation_function.parameters) if activation_function is not None else None
        self.pooling_layer = pooling_layer
        self.flatten_layer = nn.Flatten() if flatten else None

    def forward(self, x):
        x = self.conv(x)
        if self.regularization_layer is not None:
            x = self.regularization_layer(x)
        if self.activation_function is not None:
            x = self.activation_function(x)
        if self.pooling_layer is not None:
            x = self.pooling_layer(x)
        if self.flatten_layer is not None:
            x = self.flatten_layer(x)
        return x

class CNN(nn.Module):
    def __init__(self, data_shape=(3,32,32), num_classes=10, channels=[64,128,128,256,256,256,256,256,256], kernels=[3,3,3,3,3,3,3,3,3], stride = 1, padding = 'same', widths=[], pooling_layer=nn.MaxPool2d(3), pooling_positions=[5], conv_regularization_layer=None, conv_activation_function=None, mlp_regularization_layer=None, mlp_activation_function=None, headless=False, **kwargs):
        super().__init__()
        if isinstance(pooling_layer, Munch):
            pooling_layer = get_layer(pooling_layer.name, pooling_layer.parameters)
        self.conv_layers = nn.ModuleList()
        for i in range(len(channels)):
            in_channels = data_shape[0] if i == 0 else channels[i-1]
            out_channels = channels[i]
            kernel_size = kernels[i]
            pool = pooling_layer if i in pooling_positions else None
            #flatten = i == (len(channels)-1) # flatten if its the last convolutional block
            flatten = False
            self.conv_layers.append(CNNBlock(in_channels, out_channels, kernel_size, stride, padding, regularization_layer=conv_regularization_layer, activation_function=conv_activation_function, pooling_layer=pool, flatten=flatten))
                
        self.mlp_layers = nn.ModuleList()
        num_features = np.prod(self.get_layer_shape(len(channels)-1, data_shape))
        # print('num_features', num_features, self.get_layer_shape(len(channels)-1, data_shape))
        for i in range(len(widths)):
            in_channels = num_features if i == 0 else widths[i-1]
            out_channels = widths[i]
            self.mlp_layers.append(MLPBlock(in_channels, out_channels, mlp_regularization_layer, mlp_activation_function))    
        last_in = num_features if len(widths) == 0 else widths[-1]
        if not headless:
            self.mlp_layers.append(MLPBlock(last_in, num_classes, None, None)) #last layer

    def get_layer_shape(self, layer_idx, in_shape):
        with torch.no_grad():
            x = torch.rand(2, *in_shape)
            i = 0
            for layer in self.conv_layers:
                x = layer(x)
                if i == layer_idx:
                    return x.shape[1:]
                i += 1
            x = x.flatten(start_dim=1)
            for layer in self.mlp_layers:
                x = layer(x)
                if i == layer_idx:
                    return x.shape[1:]
                i += 1


    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.flatten(start_dim=1)
        for layer in self.mlp_layers:
            x = layer(x)
        return x

class VBCNN(VBModule, CNN):
    def __init__(self, data_shape=(3,32,32), num_classes=10, channels=[64,128,128,256,256,256,256,256,256], kernels=[3,3,3,3,3,3,3,3,3], stride = 1, padding = 'same', widths=[], pooling_layer=nn.MaxPool2d(3), pooling_positions=[5], conv_regularization_layer=None, conv_activation_function=None, mlp_regularization_layer=None, mlp_activation_function=None, VB_config=None, **kwargs):
        super().__init__(data_shape=data_shape, num_classes=num_classes, channels=channels, kernels=kernels, stride=stride, padding=padding, widths=widths, pooling_layer=pooling_layer, pooling_positions=pooling_positions, conv_regularization_layer=conv_regularization_layer, conv_activation_function=conv_activation_function, mlp_regularization_layer=mlp_regularization_layer, mlp_activation_function=mlp_activation_function, **kwargs)
        self.input_VB = VB_config.VB_class(data_shape, VB_config.K, VB_config.beta, VB_config=VB_config) if -1 in VB_config.positions else None
        for i in range(len(channels)+len(widths)): # CNN blocks, MLP blocks
            if i in VB_config.positions:
                self.VB_list.append(VB_config.VB_class(self.get_layer_shape(i, data_shape), VB_config.K, VB_config.beta, VB_config=VB_config))
            else:
                self.VB_list.append(None)
        self.VB_list.append(None) # Never have a VB after the final Layer


    def forward(self, x):
        # print('IN', x.shape)
        if self.input_VB is not None:
            x = self.input_VB(x)
            # print('InputVB', x.shape)
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            # print('CNN', i, x.shape)
            if self.VB_list[i] is not None:
                x = self.VB_list[i](x)
                # print('VB', x.shape)
        x = x.flatten(start_dim=1)
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
            # print('MLP', i, x.shape)
            if self.VB_list[i+len(self.conv_layers)] is not None:
                x = self.VB_list[i+len(self.conv_layers)](x)          
                # print('VB', x.shape)
        return x