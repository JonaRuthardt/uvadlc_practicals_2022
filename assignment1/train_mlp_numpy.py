################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # consider instance with maximum value as prediction of class
    predictions_indices = np.argmax(predictions, axis=1)
    targets_indices = targets

    #TODO verify correctness
    n_classes = predictions.shape[1]
    conf_mat = np.zeros((n_classes, n_classes))
    
    for pred_idx, target_idx in zip(predictions_indices, targets_indices):
      conf_mat[pred_idx, target_idx] += 1


    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    confusion_matrix = confusion_matrix.astype(np.float32)
    accuracy = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

    n_classes = confusion_matrix.shape[0]
    precision = []
    recall = []
    f1_beta = []

    for class_idx in range(n_classes):
      TP = confusion_matrix[class_idx, class_idx]
      FN = confusion_matrix[:, class_idx].sum() - TP
      FP = confusion_matrix[class_idx, :].sum() - TP

      prec = TP / (TP + FP) if TP > 0.0 else 0.0
      precision.append(prec)
      rec = TP / (TP + FN) if TP > 0.0 else 0.0
      recall.append(rec)
      f1_b = ((1.0 + beta**2) * prec * rec / (beta**2 * prec + rec)) if prec * rec > 0.0 else 0.0
      f1_beta.append(f1_b)


    metrics = {
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1_beta": f1_beta,
    }

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    metrics_list = []
    batch_size_test_list = []
    for data, labels in data_loader["test"]:
      batch_size_test = data.shape[0]
      batch_size_test_list.append(batch_size_test) 
      data = data.reshape(batch_size_test, -1) # flatten array
      # validate model on validation data
      predictions = model.forward(data)

      # evaluate results
      conf_matrix = confusion_matrix(predictions, labels)
      metrics = confusion_matrix_to_metrics(conf_matrix) 
      metrics_list.append(metrics)
    
    metrics = {k: np.average([np.mean(m[k]) for m in metrics_list], weights=batch_size_test_list) for k in metrics.keys()}

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    n_inputs = 32 * 32 * 3
    model = MLP(n_inputs=n_inputs,n_hidden=hidden_dims, n_classes=10)
    loss_module = CrossEntropyModule()

    best_model = None
    val_accuracies = []
    training_losses_total = []
    validation_losses_total = []

    for epoch in range(epochs):
      training_losses = []

      for data, labels in cifar10_loader["train"]:
        data = data.reshape(batch_size, -1) # flatten array
        # Forward training pass
        predictions = model.forward(data)
        #print(predictions)
        loss_training = loss_module.forward(predictions, labels)
        #print(loss_training)
        training_losses.append(loss_training)
        
        # Perform backprop and update weights using mini-batch SGD
        loss_derivative = loss_module.backward(predictions, labels)
        model.backward(loss_derivative)
        for layer in model.linear_layers:
          layer.params["weight"] = layer.params["weight"] - lr * layer.grads["weight"]
          layer.params["bias"] = layer.params["bias"] - lr * layer.grads["bias"]
      
      val_accuracies_epoch = []
      validation_losses = []
      batch_size_val_list = []
      for data, labels in cifar10_loader["validation"]:
        batch_size_val = data.shape[0]
        batch_size_val_list.append(batch_size_val) 
        data = data.reshape(batch_size_val, -1) # flatten array
        # validate model on validation data
        predictions = model.forward(data)
        loss_validation = loss_module.forward(predictions, labels)
        validation_losses.append(loss_validation)

        # evaluate results
        conf_matrix = confusion_matrix(predictions, labels)
        metrics = confusion_matrix_to_metrics(conf_matrix) 
        val_accuracies_epoch.append(metrics["accuracy"])
      val_accuracy = np.average(val_accuracies_epoch, weights=batch_size_val_list)
      val_accuracies.append(val_accuracy)
      if val_accuracy == max(val_accuracies):
        # save best model
        best_model = deepcopy(model)
        best_model.clear_cache()

      training_losses_total.append(np.mean(training_losses))
      validation_losses_total.append(np.average(validation_losses, weights=batch_size_val_list))
      print(f"Epoch {epoch} done; training loss: {round(np.mean(training_losses),3)}; validation loss: {round(np.average(validation_losses, weights=batch_size_val_list),3)}; validation accuracy: {round(val_accuracy,3)}")


    # TODO: Test best model
    model = best_model
    metrics = evaluate_model(model, cifar10_loader)
    test_accuracy = metrics['accuracy']
    print(f"Accuracy of best model on test set: {test_accuracy}")

    # TODO: Add any information you might want to save for plotting
    logging_info = {
      "training_losses_total": training_losses_total,
      "validation_losses_total": validation_losses_total,
    }
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here

    plot_loss_accuracy_curves = False

    if plot_loss_accuracy_curves:
      # Plot results
      import matplotlib.pyplot as plt

      fig, ax1 = plt.subplots(1,1, figsize=(10,6), dpi=120)
      ax2 = ax1.twinx()
      epochs = np.arange(1, len(val_accuracies) + 1)

      p1 = ax1.plot(epochs, logging_info["training_losses_total"], label="Training Loss", c="b")
      ax1.scatter(epochs, logging_info["training_losses_total"], c="b")
      p2 = ax1.plot(epochs, logging_info["validation_losses_total"], label="Validation Loss", c="g")
      ax1.scatter(epochs, logging_info["validation_losses_total"], c="g")
      p3 = ax2.plot(epochs, val_accuracies, label="Validation Accuracy", c="r")
      ax2.scatter(epochs, val_accuracies, c="r")
      ax1.set_xticks(epochs)

      plots = p1 + p2 + p3
      ax1.legend(plots, [p.get_label() for p in plots], loc=3)

      ax1.set_xlabel("Epoch")
      ax1.set_ylabel("Loss")
      ax2.set_ylabel("Accuracy")

      plt.savefig("numpy_plot.png")
    