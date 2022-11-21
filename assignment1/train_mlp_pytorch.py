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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


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

    predictions_indices = np.argmax(predictions, axis=1)
    targets_indices = targets

    n_classes = predictions.shape[1]
    conf_mat = np.zeros((n_classes, n_classes), dtype=np.uint32)
    
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

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    metrics_list = []

    batch_size_test_list = []
    conf_matrix_test_total = np.zeros((num_classes,num_classes), dtype=np.uint32)
    with torch.no_grad():
      for data, labels in data_loader["test"]:
        # Data preprocessing and transfer to target device
        batch_size_test_list.append(data.size()[0])
        data = data.reshape(batch_size_test_list[-1], -1)
        data = data.to(device)
        labels = labels.to(device)

        # validate model on validation data
        predictions = model(data)

        # evaluate results
        conf_matrix = confusion_matrix(predictions.numpy(), labels.numpy())
        conf_matrix_test_total += conf_matrix
        metrics = confusion_matrix_to_metrics(conf_matrix)
        metrics_list.append(metrics)

    metrics = {k: np.average([np.mean(m[k]) for m in metrics_list], weights=batch_size_test_list) for k in metrics.keys()}
    #metrics = confusion_matrix_to_metrics(conf_matrix_test_total)
    metrics["confusion_matrix_test"] = conf_matrix_test_total

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
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
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    n_inputs = 32 * 32 * 3
    model = MLP(n_inputs, hidden_dims, n_classes=10, use_batch_norm=use_batch_norm)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.to(device)

    training_losses = []
    val_accuracies = []
    best_model = None; best_accuracy = -1
    for epoch in range(epochs):
      model.train()
      training_losses.append([])
      for data, labels in cifar10_loader["train"]:

        
        # Data preprocessing and transfer to target device
        data = data.reshape(batch_size, -1)
        data = data.to(device)
        labels = labels.to(device)

        #prediction = model.forward(data)
        prediction = model(data)
        loss = loss_module(prediction, labels)
        training_losses[-1].append(loss.detach().numpy())
        
        optimizer.zero_grad()
        # Perform backpropagation
        loss.backward()
        optimizer.step()

      print(f"Training loss after epoch {epoch}: {np.mean(training_losses[-1])}")

      model.eval()
      val_accuracies.append([])
      val_batch_sizes = []
      with torch.no_grad():
        for data, labels in cifar10_loader["validation"]:
          # Data preprocessing and transfer to target device
          val_batch_sizes.append(data.size()[0])
          data = data.reshape(val_batch_sizes[-1], -1)
          data = data.to(device)
          labels = labels.to(device)

          # Predict classes
          predictions = model(data)

          # Determine metrics
          conf_matrix = confusion_matrix(predictions.numpy(), labels.numpy())
          metrics = confusion_matrix_to_metrics(conf_matrix)
          val_accuracies[-1].append(metrics["accuracy"])
        
      val_accuracy = np.average(val_accuracies[-1], weights=val_batch_sizes)
      if val_accuracy > best_accuracy:
        best_model = model.state_dict()
        best_accuracy = val_accuracy
      print(f"Validation accuracy after epoch {epoch}: {val_accuracy}")


    # Restore best performing model and test on held-out training set
    model.load_state_dict(best_model)
    metrics = evaluate_model(model, cifar10_loader)
    test_accuracy = metrics["accuracy"]
    print(f"Accuracy of best model on test set: {test_accuracy} (best validation accuracy: {best_accuracy})")
    
    logging_info = {
      "training_losses": training_losses,
      "confusion_matrix_test": metrics["confusion_matrix_test"],
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
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
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
    import matplotlib.pyplot as plt

    plot_learning_rate_plot = False
    plot_confusion_matrix = False
    plot_loss_accuracy_curves = False

    if plot_loss_accuracy_curves:
      # Plot results
      val_accuracies = [np.mean(a) for a in val_accuracies]
      training_losses = [np.mean(l) for l in logging_info["training_losses"]]

      fig, ax1 = plt.subplots(1,1, figsize=(10,6), dpi=120)
      ax2 = ax1.twinx()
      epochs = np.arange(1, len(val_accuracies) + 1)

      p1 = ax1.plot(epochs, training_losses, label="Training Loss", c="b")
      ax1.scatter(epochs, training_losses, c="b")
      p3 = ax2.plot(epochs, val_accuracies, label="Validation Accuracy", c="r")
      ax2.scatter(epochs, val_accuracies, c="r")
      ax1.set_xticks(epochs)

      plots = p1 + p3
      ax1.legend(plots, [p.get_label() for p in plots], loc=3)

      ax1.set_xlabel("Epoch")
      ax1.set_ylabel("Loss")
      ax2.set_ylabel("Accuracy")

      plt.savefig("pytorch_plot.png")

    if plot_learning_rate_plot:
      lrs = [10 ** lr for lr in range(-6,3)]
      test_accuracies = []; validation_accuracies = []; training_losses = []
      for lr in lrs:
        kwargs["lr"] = lr

        model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
        test_accuracies.append(test_accuracy)
        validation_accuracies.append(np.mean(val_accuracies))
        training_losses.append(np.mean(logging_info["training_losses"],axis=1))

      fig, ax = plt.subplots(2,1, figsize=(12,10), dpi=120)
      # plot validation accuracy / test accuarcy vs. learning rates 
      ax[0].plot(lrs, test_accuracies, label="Test Accuracy")
      ax[0].scatter(lrs, test_accuracies)
      ax[0].plot(lrs, validation_accuracies, label="Mean Validation Accuracy", c="r")
      ax[0].scatter(lrs, validation_accuracies, c="r")
      ax[0].set_xscale('log')
      ax[0].set_xticks(lrs)
      ax[0].grid()

      ax[0].legend()
      ax[0].set_xlabel("Learning Rate")
      ax[0].set_ylabel("Accuracy")

      # loss function vs. learning rates
      num_epochs = list(range(len(training_losses[0])))
      for idx, lr in enumerate(lrs):
        ax[1].plot(num_epochs, training_losses[idx], label=f"lr={lr}")
        ax[1].scatter(num_epochs, training_losses[idx])
      ax[1].grid()

      ax[1].legend()
      ax[1].set_xlabel("Epoch")
      ax[1].set_ylabel("Training Loss")

      plt.savefig("learning_rate_plot.png")
    
    if plot_confusion_matrix:
      import seaborn as sns
      conf_matrix = logging_info["confusion_matrix_test"]
      fig, ax = plt.subplots(figsize=(12,12),dpi=120)
      sns.heatmap(conf_matrix.T, annot=True, ax=ax, cmap="gray_r", fmt='g')
      ax.set_xlabel("Predicted Class")
      ax.set_ylabel("Actual Class")
      plt.yticks(rotation=0)
      plt.savefig('confusion_matrix_plot.png')

      plt.close()
      import pandas as pd
      # Plot recall, precision and f-beta scores 
      metrics = confusion_matrix_to_metrics(conf_matrix)

      data = np.array([metrics["precision"], confusion_matrix_to_metrics(conf_matrix, beta=0.1)["f1_beta"], metrics["f1_beta"], confusion_matrix_to_metrics(conf_matrix, beta=10)["f1_beta"], metrics["recall"]])
      metric_names = ["Precision", "f0.1-Score", "f1-Score", "f10-Score", "Recall"]
      df = pd.DataFrame(data, columns=[f"Class {c}" for c in range(10)], index=metric_names)

      fig, ax = plt.subplots(figsize=(12,3),dpi=120)
      sns.heatmap(df, annot=True, ax=ax, cmap="Greens")
      plt.yticks(rotation=0)
      plt.savefig("scores.png")
    