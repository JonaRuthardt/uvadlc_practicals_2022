################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # TODO: Run all hyperparameter configurations as requested

    default_hyperparameters = {
        "hidden_dims": [128],
        "use_batch_norm": False,
        "lr": 0.1,
        "batch_size": 128,
        "epochs": 20,
        "seed": 42,
        "data_dir": "data/",
    }

    results = {}

    for architecture in [[128], [256, 128], [512, 256, 128]]:

        default_hyperparameters["hidden_dims"] = architecture

        model, val_accuracies, test_accuracy, logging_info = train_mlp_pytorch.train(**default_hyperparameters)

        logging_info["val_accuracies"] = val_accuracies
        logging_info["test_accuracy"] = test_accuracy

        logging_info = {k:np.array(v).astype(float).tolist() if isinstance(v, np.ndarray) or isinstance(v, list) else v for k,v in logging_info.items()}

        results[str(architecture)] = logging_info

    with open(results_filename, 'w') as file:
        file.write(json.dumps(results))

    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    with open(results_filename, 'r') as file:
        result_dict = json.load(file)

    fig, ax = plt.subplots(2,1, figsize=(10,10), dpi=120)
    epochs = np.arange(1, 20 + 1)
    ax[0].set_xticks(epochs)
    ax[1].set_xticks(epochs)
    ax[0].set_xlabel("Epoch")
    ax[1].set_xlabel("Epoch")
    ax[0].set_ylabel("Training Loss")
    ax[1].set_ylabel("Validation Accuracy")

    configurations = []
    for hyperparameter, config_results in result_dict.items():
        configurations.append(hyperparameter)
        configurations.append(None)

        val_accuracies = [np.mean(a) for a in config_results["val_accuracies"]]
        training_losses = [np.mean(l) for l in config_results["training_losses"]]

        p1 = ax[0].plot(epochs, training_losses)
        ax[0].scatter(epochs, training_losses)
        p3 = ax[1].plot(epochs, val_accuracies)
        ax[1].scatter(epochs, val_accuracies)
        
    ax[0].legend(configurations)
    ax[1].legend(configurations)

        

    plt.savefig("architecture_plot.png")

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'results_question_2-5.txt' 
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)