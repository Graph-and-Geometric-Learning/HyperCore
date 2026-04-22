"""
Lorentz Batch Normalization Layers.

These modules implement batch normalization in Lorentzian (hyperbolic) geometry,
using Lorentzian centroids and Fréchet variance for centering and scaling. 
Optionally, can compute the batch norm om spatial coordiantes instead.

Based on:
    - Lorentzian Residual Neural Networks (https://arxiv.org/abs/2412.14695)
    - Fully Hyperbolic Convolutional Neural Networks for Computer Vision (https://arxiv.org/abs/2303.15919)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoopt import ManifoldParameter
from ...manifolds import Lorentz

class LorentzBatchNorm(nn.Module):
    """
    Lorentz Batch Normalization with Centroid and Fréchet Variance.

    This normalization shifts the batch to a common Lorentzian centroid, rescales
    using the Fréchet variance, and shifts to a learnable mean point.

    Args:
        manifold_in (Lorentz): Lorentz manifold for input.
        num_features (int): Number of spatial dimensions (excluding time coordinate).
        manifold_out (Lorentz, optional): Optional target manifold for output projection.
    """
    def __init__(self, manifold_in: Lorentz, num_features: int, manifold_out=None):
        super(LorentzBatchNorm, self).__init__()
        self.manifold = manifold_in
        self.c = manifold_in.c
        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))

        self.manifold_out = manifold_out
        self.eps = 1e-5

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones((1,)))

    def forward(self, x, momentum=0.1):
        """
        Forward pass of Lorentz BatchNorm.

        Args:
            x (torch.Tensor): Input tensor [batch_size, ..., features].
            momentum (float): Momentum for running statistics update.

        Returns:
            torch.Tensor: Normalized output tensor.
        """
        assert (len(x.shape)==2) or (len(x.shape)==3), "Wrong input shape in Lorentz batch normalization."
        beta = self.beta
        if self.training:
            mean = self.manifold.lorentzian_centroid(x)
            if len(x.shape) == 3:
                mean = self.manifold.lorentzian_centroid(mean)
            # Transport batch to origin (center batch)
            x_T = self.manifold.logmap(mean, x)
            x_T = self.manifold.transp0back(mean, x_T)

            # Compute Fréchet variance
            if len(x.shape) == 3:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=(0,1))
            else:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=0)

            # Rescale batch
            x_T = x_T*(self.gamma/(var+self.eps))
            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)
            # Save running parameters
            with torch.no_grad():
                running_mean = self.manifold.expmap0(self.running_mean)
                means = torch.concat((running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(self.manifold.logmap0(self.manifold.lorentzian_centroid(means, weight=torch.tensor(((1-momentum), momentum), device=means.device))).reshape(self.running_mean.shape))
                self.running_var.copy_((1 - momentum)*self.running_var + momentum*var.detach())

        else:
            # Transport batch to origin (center batch)
            running_mean = self.manifold.expmap0(self.running_mean)
            x_T = self.manifold.logmap(running_mean, x)
            x_T = self.manifold.transp0back(running_mean, x_T)

            # Rescale batch
            x_T = x_T*(self.gamma/(self.running_var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)
        
        if self.manifold_out is not None:
            output = output * (self.manifold_out.c / self.c).sqrt()

        return output

class LorentzBatchNorm1d(LorentzBatchNorm):
    """
    1D Lorentz Batch Normalization with Centroid and Fréchet variance.

    Optionally, if `space_method` is enabled, applies standard Euclidean
    BatchNorm1d to the spatial components instead of hyperbolic normalization.

    Args:
        manifold_in (Lorentz): Lorentz manifold for input.
        num_features (int): Number of spatial features (excluding time).
        manifold_out (Lorentz, optional): Target manifold for output projection.
        space_method (bool, optional): If True, apply Euclidean BatchNorm on spatial part.
        momentum (float, optional): Momentum for running statistics. Default is 0.1.
    """
    def __init__(self, manifold_in: Lorentz, num_features: int, manifold_out=None, space_method=False, momentum=0.1):
        super(LorentzBatchNorm1d, self).__init__(manifold_in, num_features, manifold_out)
        if space_method:
            self.norm = nn.BatchNorm1d(num_features=num_features, momentum=momentum)
    def forward(self, x, momentum=0.1):
        """
        Forward pass for LorentzBatchNorm1d.

        Args:
            x (torch.Tensor): Input tensor [batch_size, features].
            momentum (float, optional): Optional momentum override.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        if momentum is not None:
            self.momentum=momentum
        if self.space_method:
            x_space = x[..., 1:]
            normed_space = self.norm(x_space)
            x_time = ((normed_space ** 2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
            x = torch.cat([x_time, normed_space], dim=-1)
            if self.manifold_out is not None:
                x = x * (self.manifold_out.c / self.c).sqrt()
            return x
        return super(LorentzBatchNorm1d, self).forward(x, momentum)
    
class LorentzBatchNorm2d(LorentzBatchNorm):
    """
    2D Lorentz Batch Normalization with Centroid and Fréchet variance.

    Input is expected in **channels-last** format: [batch_size, height, width, channels].

    Optionally, if `space_method` is enabled, applies standard Euclidean
    BatchNorm2d on spatial components instead of hyperbolic normalization.

    Args:
        manifold_in (Lorentz): Lorentz manifold for input.
        num_channels (int): Number of channels (features).
        manifold_out (Lorentz, optional): Target manifold for output projection.
        space_method (bool, optional): If True, apply Euclidean BatchNorm on spatial part.
        momentum (float, optional): Momentum for running statistics. Default is 0.1.
    """
    def __init__(self, manifold_in: Lorentz, num_channels: int, manifold_out=None, space_method=False, momentum=0.1):
        super(LorentzBatchNorm2d, self).__init__(manifold_in, num_channels, manifold_out)
        self.space_method = space_method
        if space_method:
            self.norm = nn.BatchNorm2d(num_features=num_channels, momentum=momentum)
        self.momentum=momentum
    def forward(self, x, momentum=None):
        """
        Forward pass for LorentzBatchNorm2d.

        Args:
            x (torch.Tensor): Input tensor [batch_size, height, width, channels].
            momentum (float, optional): Optional momentum override.

        Returns:
            torch.Tensor: Normalized tensor with same shape.
        """
        
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        if momentum is not None:
            self.momentum=momentum
        if self.space_method:
            x = x.permute(bs, c, h, w)
            x_space = x[..., 1:]
            normed_space = self.norm(x_space)
            x_time = ((normed_space ** 2).sum(dim=-1, keepdims=True) + self.c).clamp_min(1e-6).sqrt()
            x = torch.cat([x_time, normed_space], dim=-1)
            if self.manifold_out is not None:
                x = x * (self.manifold_out.c / self.c).sqrt()
            return x.reshape(bs, h, w, c)
        x = x.view(bs, -1, c)
        x = super(LorentzBatchNorm2d, self).forward(x, self.momentum)
        x = x.reshape(bs, h, w, c)
        return x
