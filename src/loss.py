import logging

import torch

from src.models.modules import VBModule


def get_loss_function(loss, model=None, ignore_layers=[], *args, **kwargs):
    if model is not None and hasattr(model, 'base_model'):
        model = model.base_model
    if loss == 'CrossEntropy':
        loss_function = CrossEntropy()
    elif 'VB' in loss:
        assert isinstance(model, VBModule), 'If you are using VB loss you need to use a VBModule based model.'
        loss_function = VBLoss(model)
    elif loss == 'GL2':
        loss_function = GL2(ignore_layers)
    elif loss == 'GCosineDistance':
        loss_function = GCosineDistance(ignore_layers)
    else:
        raise ValueError(f'The loss function {loss} is not implemented yet.')
    logging.info(f'Using {loss} as loss function.')
    return loss_function



def TV(x):
    """Anisotropic TV."""
    if len(x.shape) == 3: #single image
        dx = torch.mean(torch.abs(x[:, :, :-1] - x[:, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :-1, :] - x[:, 1:, :]))
    else:
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy
    

class CrossEntropy:
    def __init__(self):
        self.nn_ce = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        self.format = '.5f'
        self.name = 'CrossEntropy'
        self.subject_to = 'min'

    def __call__(self, prediction, targets):
        if prediction.shape == targets.shape:
            ce = - torch.mean(torch.sum(torch.softmax(targets, -1) * torch.log(torch.softmax(prediction, -1)), dim=-1))
        else:
            ce = self.nn_ce(prediction, targets)
        return ce

class VBLoss:
    def __init__(self, model):
        self.model_loss = model.loss
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        self.format = '.5f'
        self.name = 'VBLoss'
        self.subject_to = 'min'
        
    def __call__(self, prediction, targets):
        vb_loss = self.model_loss()
        ce_loss = self.ce_loss(prediction, targets)
        return vb_loss + ce_loss

class GCosineDistance:
    def __init__(self, ignore_layers=[]):
        super().__init__()
        self.format = '.6f'
        self.name = 'GCosineDistance'
        self.subject_to = 'min'
        self.ignore_layers = ignore_layers

    def __call__(self, prediction, target):
        total_loss = 0
        p_norm = 0
        t_norm = 0
        for layer, ((p_name, p_grad), (t_name, t_grad)) in enumerate(zip(prediction.items(), target.items())):
            assert p_name == t_name, f'Layer names for gradients do not match at layer {layer}!. {p_name} != {t_name} !'
            if any(n in p_name for n in self.ignore_layers):
                continue
            partial_loss = (p_grad * t_grad).sum()
            partial_p_norm = p_grad.pow(2).sum()
            partial_t_norm = t_grad.pow(2).sum()
            
            total_loss += partial_loss
            p_norm += partial_p_norm
            t_norm += partial_t_norm
        total_loss = 1 - total_loss / ((p_norm.sqrt()*t_norm.sqrt())+1e-8)
        return total_loss

class GL2:
    def __init__(self, ignore_layers=[]):
        super().__init__()
        self.base_loss_fn = torch.nn.MSELoss(size_average=None, reduction='sum')
        self.format = '.6f'
        self.name = 'GL2'
        self.subject_to = 'min'
        self.ignore_layers = ignore_layers

    def __call__(self, prediction, target):
        total_loss = 0
        for layer, ((p_name, p_grad), (t_name, t_grad)) in enumerate(zip(prediction.items(), target.items())):
            assert p_name == t_name, f'Layer names for gradients do not match at layer {layer}!. {p_name} != {t_name} !'
            if any(n in p_name for n in self.ignore_layers):
                continue
            partial_loss = self.base_loss_fn(p_grad, t_grad)
            total_loss += partial_loss
        return total_loss