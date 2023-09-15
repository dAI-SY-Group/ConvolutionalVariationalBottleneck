from copy import deepcopy
from numbers import Number

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

def prepare_VB_config(VB_config):
    assert VB_config is not None, 'VB_config is None but needs to be provided to create a VBModule'
    VB_config = deepcopy(VB_config)
    VB_class = eval(VB_config._class)
    VB_config.pop('_class')
    VB_config.VB_class = VB_class    
    return VB_config

"""
    Implementation inspired by https://github.com/1Konny/VIB-pytorch/blob/master/model.py
    
    separated into more submodules
"""

class EncoderBase(nn.Module):
    def __init__(self, in_shape, K=256, beta=1e-3, bias=False, *args, **kwargs):
        super(EncoderBase, self).__init__()
        self.in_shape = in_shape
        self.K = K
        self.beta = beta
        self.bias = bias

    def reparameterize(self, mu, std):
        def check_number(vector):
            if isinstance(vector, Number):
                return torch.Tensor([vector])
            else:
                return vector
        mu = check_number(mu)
        std = check_number(std)
        eps = Variable(std.data.new(std.size()).normal_().to(mu.device))
        return mu + eps * std

    def loss(self):
        return self.beta * torch.mean(-0.5 * torch.sum(1 + self.std - self.mu ** 2 - self.std.exp(), dim = 1)) # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    

class ConvolutionalVariationalEncoder(EncoderBase):
    def __init__(self, in_shape, K=256, beta=1e-3, bias=False, *args, **kwargs):
        super(ConvolutionalVariationalEncoder, self).__init__(in_shape=in_shape, K=K, beta=beta, bias=bias, **kwargs)
        VB_cfg = kwargs.get('VB_config', None)
        if VB_cfg is not None:
            self.kernel_size = VB_cfg.get('kernel_size', 3)
        else:
            self.kernel_size = 3

        self.mu_encoder = nn.Conv2d(self.in_shape[0], self.K, self.kernel_size, bias=self.bias, padding='same')
        self.std_encoder = nn.Conv2d(self.in_shape[0], self.K, self.kernel_size, bias=self.bias, padding='same')

        # shape of the feature outputs as they now capture the statistics
        self.mu = Variable(torch.zeros(self.K, *self.in_shape[1:]))
        self.std = Variable(torch.zeros(self.K, *self.in_shape[1:]))

    def forward(self, x):
        mu = self.mu_encoder(x)
        std = F.softplus(torch.clip(self.std_encoder(x), -5, 5), beta=1) # start similar to e^0.5x just more linear for higher values -> dont let stds get too high, clip to keep them more bounded

        self.mu = mu
        self.std = std

        encoding = self.reparameterize(mu, std) # pull sample from the distribution

        return encoding
    
class ConvolutionalVariationalDecoder(nn.Module):
    def __init__(self, in_shape, K=256, bias=False, *args, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        self.K = K
        self.bias = bias

        self.decoder = nn.Conv2d(self.K, in_shape[0], 1, bias=self.bias)

    def forward(self, encoding):
        x = self.decoder(encoding)
        return x
    

class FullyConvolutionalVariationalBottleneck(nn.Module):
    def __init__(self, in_shape, K=256, beta=1e-3, bias=False, *args, **kwargs):
        super().__init__()
        self.in_shape = in_shape
        if K == 'adaptive':
            # half of channel dimension of in shape if >=8, else 8
            if in_shape[0] >= 8:
                self.K = in_shape[0] // 2
            else:
                self.K = 8
        else:
            self.K = K
        self.beta = beta
        self.bias = bias

        self.encoder = ConvolutionalVariationalEncoder(in_shape, K=self.K, beta=self.beta, bias=self.bias, **kwargs)
        self.decoder = ConvolutionalVariationalDecoder(in_shape, K=self.K, bias=self.bias, **kwargs)

        self.out_feats = Variable(torch.zeros(in_shape))

    def forward(self, x):
        encoding = self.encoder(x)
        self.out_feats = self.decoder(encoding)
        return self.out_feats

    def loss(self):
        return self.encoder.loss() 

class VBModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.input_VB = None
        self.VB_list = nn.ModuleList()

    def loss(self):
        loss = torch.zeros((1,)).cuda()
        if self.input_VB is not None:
            loss = torch.add(loss, self.input_VB.loss())
        for vb in self.VB_list:
            if type(vb) in [FullyConvolutionalVariationalBottleneck]:
                loss = torch.add(loss, vb.loss())
                #print(f'VB loss: {loss}')
        return loss
    
    def get_vb_layers(self, positions=None):
        if positions is not None:
            vb_layers = [vb for i, vb in enumerate(self.VB_list) if i in positions and vb is not None]
        else: 
            vb_layers = [vb for vb in self.VB_list if vb is not None]
        if self.input_VB is not None:
            vb_layers = [self.input_VB] + vb_layers
        return vb_layers

class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, regularization_layer=None, activation_function=None, use_bias=True, dense=nn.Linear):
        super().__init__()
        self.linear = dense(in_channels, out_channels, bias=use_bias)
        self.regularization_layers = nn.ModuleList()
        if not isinstance(regularization_layer, list):
            regularization_layer = [regularization_layer]
        for reg_layer in regularization_layer:
            if reg_layer is not None:
                if 'BatchNorm' in reg_layer.name:
                    assert reg_layer.name == 'nn.BatchNorm1d', 'If you want to use BatchNorm for Linear layer. you need to explicitly use BatchNorm1d!'
                self.regularization_layers.append(get_layer(reg_layer.name, reg_layer.parameters, channels = out_channels))
        self.activation_function = get_layer(activation_function.name, activation_function.parameters) if activation_function is not None else None

    def forward(self, x):
        x = self.linear(x)
        for reg_layer in self.regularization_layers:
            x = reg_layer(x)
        if self.activation_function is not None:
            x = self.activation_function(x)
        return x

class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=2):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x

class MaskDropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__(p, inplace)
        self.do_mask = None
        self.mask_shape = None
        self.use_mask = False
        self.track_mask = False
        self.p = p

    def forward(self, x):
        if self.use_mask and self.do_mask is not None and not self.track_mask:
            return x * self.do_mask * (1/(1-self.p))
        else:
            x = super().forward(x)
            if self.track_mask:
                self.do_mask = (x != 0)
                self.mask_shape = self.do_mask.shape
                self.track_mask = False
            return x


"""This is code for median pooling from https://gist.github.com/rwightman.

https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598
"""
class MedianPool2d(nn.Module):
    """Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=True):
        """Initialize with kernel_size, stride, padding."""
        super().__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
    
def get_layer(layer_str, parameters, force_parameters=False, **kwargs):
    """
    Get a neural network layer.

    This function takes a layer string (e.g., nn.ReLU, nn.Linear) and the corresponding parameters 
    required for that module.

    Args:
        layer_str (str): The string representation of the layer (e.g., 'nn.ReLU').
        parameters (list): List of parameters needed for the specified layer.
        force_parameters (bool, optional): If True, force the use of provided parameters, 
            otherwise use default parameters. Defaults to False.
        **kwargs: Additional keyword arguments that can be used to pass specific parameters 
            required by certain layers (e.g., 'channels' for BatchNorm and InstanceNorm layers).

    Returns:
        torch.nn.Module: The specified layer module.

    Raises:
        NameError: If the provided layer string is not a valid module.

    Example:
        >>> get_layer('nn.Linear', [1024, 1024])
        Linear(in_features=1024, out_features=1024, bias=True)

    Note:
        - For BatchNorm and InstanceNorm layers, if `force_parameters` is not set to True,
          it will use 'channels' from keyword arguments as parameters.

    """
    if 'BatchNorm' in layer_str or 'InstanceNorm' in layer_str and not force_parameters:
        parameters = [kwargs['channels']]
    return eval(f'{layer_str}(*parameters)')
