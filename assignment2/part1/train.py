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
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from torch.utils.data import DataLoader

from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False

    # Randomly initialize and modify the model's last layer for CIFAR100.
    model.fc = nn.LazyLinear(num_classes)
    #TODO correct approach to do it like that?

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_dataset, val_dataset = get_train_validation_set(data_dir, augmentation_name=augmentation_name)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_module = nn.CrossEntropyLoss()

    # Training loop with validation after each epoch. Save the best model.
    best_accuracy = -1
    best_model = None
    for epoch in range(epochs):
        # Training
        model.train()
        counter = 0
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) #TODO is use of dataloader like this valid?
        for data, labels in train_dataloader:
            data = data.to(device)
            labels = labels.to(device)
            prediction = model(data)
            #_, prediction = torch.max(prediction, 1)
            loss = loss_module(prediction, labels)
            #TODO which loss to use?

            # Perform backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            counter += 1
            if counter % len(train_dataset) // batch_size // 100 == 0:
                print(f"{len(train_dataset) // batch_size // counter}% done with training")

        # Evaluation
        
        validation_accuracy = evaluate_model(model, val_dataloader, device)
        print(f"Validation Accuracy after Epoch {epoch}: {validation_accuracy}")
        if validation_accuracy > best_accuracy:
            best_model = model.state_dict()
            best_accuracy = validation_accuracy
            #TODO do checkpoints need to be stored?

    # Load the best model on val accuracy and return it.
    model.load_state_dict(best_model)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    accuracies = []
    batch_sizes = []
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            batch_sizes.append(data.size()[0])

            prediction = model(data)
            _, predicted_class = torch.max(prediction, dim=1)
            accuracies.append(sum([1 if pred == gt else 0 for pred, gt in zip(predicted_class, labels)]) / batch_sizes[-1])
            
    accuracy = np.average(accuracies, weights=batch_sizes)

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Connected to GPU")
    else:
        #raise ValueError("No GPU available")
        print("WARNING: no GPU")
        device = torch.device("cpu")

    # Load the model
    model = get_model()
    model.to(device)

    # Get the augmentation to use
    pass

    # Train the model
    checkpoint_name = None #TODO implement properly
    train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name)

    # Evaluate the model on the test set
    test_dataset = get_test_set(data_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    test_accuracy = evaluate_model(model, test_dataloader, device)
    print(f"Test accuracy was {round(test_accuracy, 3)}")

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
