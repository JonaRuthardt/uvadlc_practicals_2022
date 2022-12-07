################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    z = mean + std * torch.randn_like(std)
    
    #######################
    # END OF YOUR CODE    #
    #######################
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    KLD = 0.5 * torch.sum(torch.exp(2 * log_std) + mean**2 - 1 - 2 * log_std, dim=-1)

    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #TODO remove when done
    
    #######################
    # END OF YOUR CODE    #
    #######################
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    # print(elbo.size())
    elbo = torch.sum(elbo)
    #elbo = torch.mean(elbo) #TODO mean or sum?
    bpd = elbo * np.log2(np.e) / np.prod(img_shape)


    
    #######################
    # END OF YOUR CODE    #
    #######################
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    grid_values = torch.linspace(0.5/grid_size, (grid_size-0.5)/grid_size, grid_size)
    #grid_values = torch.tensor([0.5 * i / grid_size for i in range(1, grid_size+1)])
    grid = torch.meshgrid(grid_values, grid_values)

    samples = torch.distributions.Normal(0, 1).icdf(torch.stack(grid, dim=-1).reshape(-1, 2)) #TODO check if this is correct

    prediction = decoder(samples)
    prediction = torch.softmax(prediction, dim=-1)
    #TODO do it this way? https://piazza.com/class/l8vanaiu2bh1rg/post/226
    # sample the pixels of the image from a categorical distribution whose parameters are given by the output of the decoder (after applying the softmax).
    # use torch.multinomial to sample from a categorical distribution.
    # prediction = torch.multinomial(prediction, num_samples=1, replacement=True)

    img_grid = make_grid(prediction, nrow=grid_size)
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return img_grid

