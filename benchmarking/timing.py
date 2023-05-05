import torch
from torch import nn
import time


def inference(model, features, labels):
    """
    Times one forward pass of a model with the first feature and label.

    Args
    ____
    model : nn.Module
        Model for testing
    features : torch.Tensor
        Input features tensor of shape ``(m, n)``
    labels : torch.Tensor
        Labels tensor of shape ``(m)``

    Returns
    _______
    total_time : float
        Total time for forward pass
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=0.3)  # initialize optimizer
    loss_func = nn.BCELoss(reduction='mean')  # initialize loss function

    model.eval()  # Evaluate the model on the training set

    total_time = 0

    X_batch = torch.tensor(features[0])
    y_batch = torch.tensor(labels[0]).to(torch.float32)

    start_time = time.time()
    y_pred = model(X_batch)
    finish_time = time.time()

    total_time += (finish_time - start_time)
    return total_time


def train(model, features, labels):
    """
    Times one training pass of a model with the first feature and label.

    Args
    ____
    model : nn.Module
        Model for testing
    features : torch.Tensor
        Input features tensor of shape ``(m, n)``
    labels : torch.Tensor
        Labels tensor of shape ``(m)``

    Returns
    _______
    total_time : float
        Total time for training pass
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.3)  # initialize optimizer
    loss_func = nn.BCELoss(reduction='mean')  # initialize loss function

    model.train()  # Train the model on the training set

    train_loss = 0
    total_time = 0

    X_batch = torch.tensor(features[0])
    y_batch = torch.tensor(labels[0]).to(torch.float32)

    start_time = time.time()
    y_pred = model(X_batch)
    loss = loss_func(y_pred, y_batch)
    train_loss += loss.item()
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    finish_time = time.time()

    total_time += (finish_time - start_time)
    return total_time