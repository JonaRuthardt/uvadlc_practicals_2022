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
from dataset import *
import clip


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
    #model.fc = nn.LazyLinear(num_classes)
    model.fc = nn.Linear(512, num_classes)
    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.fill_(0)

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


def main(args):
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
    set_seed(args.seed)

    # Set the device to use for training
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Connected to GPU", flush=True)
    else:
        raise ValueError("No GPU available")


    # Load the model
    model = get_model()
    #model = model._load_from_state_dict()
    model.load_state_dict(torch.load("/home/lcur0653/uvadlc_practicals_2022-1/best_resnet18_model.model"))
    model.to(device)

    # Evaluate the model on the test set
    _, preprocess = clip.load(args.arch)
    #dataset = load_dataset(args.dataset, args.root, args.split, preprocess)
    train_dataset, val_dataset, test_dataset = load_dataset(
            args, preprocess
        )
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    test_accuracy = evaluate_model(model, test_dataloader, device)
    print(f"Test accuracy was {round(test_accuracy, 3)}", flush=True)

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument("--root", type=str, default="./data", help="dataset")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--dataset", type=str, default="cifar100", help="dataset")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument(
        "--test_noise",
        default=False,
        action="store_true",
        help="whether to add noise to the test images",
    )
    parser.add_argument("--arch", type=str, default="ViT-B/32")

    args = parser.parse_args()
    kwargs = vars(args)
    main(args)
