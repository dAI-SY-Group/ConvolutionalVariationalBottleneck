from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models import resnet
from munch import munchify       

from src.models.modules import GroupNorm, VBModule, MaskDropout, prepare_VB_config


def get_resnet_config():
    #norm: None == BatchNorm!
    return munchify({
    'ResNet18GN': {'builder': ResNet, 'norm': 'GN', 'load_identifier': 18, 'kwargs': {'block': BasicBlock, 'layers': [2,2,2,2]}},
    'VBResNet18GN': {'builder': VBResNet, 'norm': 'GN', 'load_identifier': 18, 'kwargs': {'block': BasicBlock, 'layers': [2,2,2,2]}},
    })


def build_ResNet(architecture, config):
    rn_config = get_resnet_config()[architecture]
    builder = rn_config.builder
    num_classes = config.num_classes
    dropout_rate = 0 if config is None or config.model.dropout_rate is None else config.model.dropout_rate
    if rn_config.norm == 'GN':
        gn_groups = 2 if config == None else config.model.gn_groups
        norm_layer = lambda ch: GroupNorm(ch, gn_groups)
    else:
        norm_layer = None #BatchNorm!

    if 'VB' in architecture:
        vb_kwargs = {'in_shape': config.data.shape, 'VB_config': prepare_VB_config(config.model.VB)}
    else:
        vb_kwargs = {}

    model = builder(num_classes = num_classes, norm_layer=norm_layer, dropout_rate=dropout_rate, **rn_config.kwargs, **vb_kwargs)
    
    return model



#PyTorch ResNet wrapper to introduce **kwargs for VB Modules
class ResNet(resnet.ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_rate = 0,
        **kwargs
    ) -> None:
        super(ResNet, self).__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.dropout = MaskDropout(dropout_rate)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def init_VBResNet(blocks, in_shape, base_width, VB_config=None):
    VB_list = nn.ModuleList()
    sides = in_shape[1]// 4
    width = base_width
    for i in range(blocks): # number of blocks + 1 possible position after pooling
        if i == blocks-1:
            width = width // 2
        if i in VB_config.positions:
            if i < blocks-1:
                m = VB_config.VB_class((width, sides, sides), VB_config.K, VB_config.beta, VB_config=VB_config)
            else:
                m = VB_config.VB_class((width,), VB_config.K, VB_config.beta, VB_config=VB_config)
        else:
            m = nn.Identity()
        VB_list.append(m)
        sides = sides // 2
        width = width * 2 #if i < blocks-3 else width * block_exp
    return VB_list

class VBResNet(VBModule, ResNet):
    def __init__(
        self,
        block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        in_shape=(3,32,32), VB_config=None, **kwargs
    ) -> None:
        super().__init__(block=block, layers=layers, num_classes=num_classes, zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, **kwargs)
        self.input_VB = VB_config.VB_class(in_shape, VB_config.K, VB_config.beta, VB_config=VB_config) if -1 in VB_config.positions else None
        self.VB_list = init_VBResNet(5, in_shape, width_per_group, VB_config=VB_config)

    def _forward_impl(self, x):
        if self.input_VB is not None:
            x = self.input_VB(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if self.VB_list[0] is not None:
            x = self.VB_list[0](x)
        x = self.layer2(x)
        if self.VB_list[1] is not None:
            x = self.VB_list[1](x)
        x = self.layer3(x)
        if self.VB_list[2] is not None:
            x = self.VB_list[2](x)
        x = self.layer4(x)
        if self.VB_list[3] is not None:
            x = self.VB_list[3](x)
        x = self.avgpool(x)
        if self.VB_list[4] is not None:
            x = self.VB_list[4](x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x