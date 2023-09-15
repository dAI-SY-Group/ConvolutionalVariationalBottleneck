import os
import logging

import torch
import numpy as np
import torchmetrics
from torchmetrics import Metric

from src.utils import build_config, Normalizer


IMPLEMENTED_METRICS = [ 'MSE',
                        'PSNR', 'SSIM',
                        'LPIPS',
                        'Accuracy'
                        ]




def get_metric_function(metric, *args, **kwargs):
    """
    Get the metric function based on the provided metric configuration.

    This function initializes and configures a metric function based on the provided metric
    configuration. If a string is provided, it attempts to load the metric configuration
    from the local configurations. It then checks if the metric is implemented and either
    uses a pre-defined metric from torchmetrics or evaluates the metric using a custom
    implementation.

    Args:
        metric (str or Munch): 
            The metric to be used. It can be either a string specifying the metric name
            or a Munch object containing the metric configuration.
        *args: 
            Variable length argument list.
        **kwargs: 
            Arbitrary keyword arguments.

    Returns:
        Callable: 
            The initialized metric function.

    Raises:
        AssertionError: 
            If the specified metric is not implemented or checked.

    Example:
        metric_function = get_metric_function(metric, *args, **kwargs)

    Note:
        - `metric` can be either a string or a Munch based metric configuration.

    """
    if isinstance(metric, str): #if only a string was given try to fetch the metric config from local configurations
        # find 'metric_config_paths.yaml in the directory of this source file
        # and load the metric config from there
        metric_paths = build_config(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'configs', 'metrics', '_paths.yaml'))
        metrics_dict = build_config(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'configs', 'metrics', metric_paths[metric]+'.yaml'))
        metric = metrics_dict.metrics[metric]
        
    assert metric.name in IMPLEMENTED_METRICS, f'The metric function {metric.name} is not yet implemented or checked.'
    metric_args = metric.metric_args if metric.metric_args else {} #because some torch metrics dont need any arguments
    if metric.own_implementation:
        metric_function = eval(metric.name)(**metric_args, **kwargs)
    else:
        metric_function = eval('torchmetrics.'+metric.tm_name)(**metric_args, **kwargs)
    
    #add functionality to the metric function
    metric_function.name = metric.name
    metric_function.format = metric.format
    metric_function.subject_to = metric.subject_to

    logging.info(f'Using {metric_function.name} as metric function.')
    if 'device' in kwargs:
        metric_function.to(kwargs['device'])
    return metric_function

    

    
class LPIPS(torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity):
    """
    LPIPS metric with optional normalization.

    This metric calculates the Learned Perceptual Image Patch Similarity (LPIPS) using
    the torchmetrics implementation. It optionally normalizes the inputs before calculation.

    Args:
        reduction (str): 
            Reduction method. Default is 'elementwise_mean'.
        scale_01 (bool): 
            Flag to indicate whether to scale inputs to the range [0, 1]. Default is True.
        *args: 
            Variable length argument list.
        **kwargs: 
            Arbitrary keyword arguments.

    Example:
        lpips = LPIPS()
        lpips.update(predictions, targets)
        similarity = lpips.compute()

    Note:
        - This metric extends the `torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity` class.
        - Optionally, it normalizes inputs before calculation.

    See Also:
        - `torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity` for details on LPIPS.

    """
    def __init__(self, reduction='elementwise_mean', scale_01=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = Normalizer(mode='single') if scale_01 else None

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.normalizer:
            preds = self.normalizer(preds)
            target = self.normalizer(target)
        #If the input is a single channel image, repeat it 3 times to match the expected input of the lpips network
        if preds.shape[1] == 1:
            preds = preds.repeat(1,3,1,1)
        if target.shape[1] == 1:
            target = target.repeat(1,3,1,1)
        return super().update(preds, target)

class SSIM(torchmetrics.StructuralSimilarityIndexMeasure):
    """
    Structural Similarity Index Measure (SSIM) metric with optional normalization.

    This metric calculates the SSIM using the torchmetrics implementation. It optionally
    normalizes the inputs before calculation.

    Args:
        scale_01 (bool): 
            Flag to indicate whether to scale inputs to the range [0, 1]. Default is True.
        *args: 
            Variable length argument list.
        **kwargs: 
            Arbitrary keyword arguments.

    Example:
        ssim = SSIM()
        ssim.update(predictions, targets)
        similarity = ssim.compute()

    Note:
        - This metric extends the `torchmetrics.StructuralSimilarityIndexMeasure` class.
        - Optionally, it normalizes inputs before calculation.

    See Also:
        - `torchmetrics.StructuralSimilarityIndexMeasure` for details on SSIM.

    """
    def __init__(self, scale_01=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = Normalizer(mode='single') if scale_01 else None        
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if self.normalizer:
            preds = self.normalizer(preds)
            target = self.normalizer(target)
        return super().update(preds, target)

class PSNR(torchmetrics.PeakSignalNoiseRatio):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric with optional normalization.

    This metric calculates the PSNR using the torchmetrics implementation. It optionally
    normalizes the inputs before calculation.

    Args:
        scale_01 (bool): 
            Flag to indicate whether to scale inputs to the range [0, 1]. Default is True.
        *args: 
            Variable length argument list.
        **kwargs: 
            Arbitrary keyword arguments.

    Example:
        psnr = PSNR()
        psnr.update(predictions, targets)
        psnr_value = psnr.compute()

    Note:
        - This metric extends the `torchmetrics.PeakSignalNoiseRatio` class.
        - Optionally, it normalizes inputs before calculation.

    See Also:
        - `torchmetrics.PeakSignalNoiseRatio` for details on PSNR.

    """
    def __init__(self, scale_01=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = Normalizer(mode='single') if scale_01 else None        
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if self.normalizer:
            preds = self.normalizer(preds)
            target = self.normalizer(target)
        return super().update(preds, target)

class MSE(torchmetrics.MeanSquaredError):
    """
    Mean Squared Error (MSE) metric with optional normalization.

    This metric calculates the MSE using the torchmetrics implementation. It optionally
    normalizes the inputs before calculation.

    Args:
        scale_01 (bool): 
            Flag to indicate whether to scale inputs to the range [0, 1]. Default is True.
        *args: 
            Variable length argument list.
        **kwargs: 
            Arbitrary keyword arguments.

    Example:
        mse = MSE()
        mse.update(predictions, targets)
        mse_value = mse.compute()

    Note:
        - This metric extends the `torchmetrics.MeanSquaredError` class.
        - Optionally, it normalizes inputs before calculation.

    See Also:
        - `torchmetrics.MeanSquaredError` for details on MSE.

    """
    def __init__(self, scale_01=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = Normalizer(mode='single') if scale_01 else None        
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if self.normalizer:
            preds = self.normalizer(preds)
            target = self.normalizer(target)
        return super().update(preds, target)
