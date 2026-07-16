import torch
from torch import nn
from tqdm import tqdm

from .evaluation import evaluate_loss


def train_model(
    net,
    optimizer,
    train_data,
    val_data,
    num_epochs=10,
    label="Model",
):
    """
    Train a model and record validation loss after each epoch.

    Args:
        net : SNNNetwork
            Network to train.

        optimizer : torch.optim.Optimizer
            Optimiser used to update the network parameters.

        train_data : callable
            Function returning a fresh generator of training batches.

        val_data : callable
            Function returning a fresh generator of validation batches.

        num_epochs : int
            Number of training epochs.

        label : str
            Model name shown in the progress bar.

    Returns:
        net : SNNNetwork
            Trained network.

        hist : list
            Validation loss history.
    """

    criterion = nn.MSELoss()  # MSE loss
    hist = []  # validation loss history

    for epoch in tqdm(
        range(num_epochs),
        desc=f"Training {label}",
    ):
        net.train()

        # Create a fresh training-data generator
        gen = train_data()

        for x, y in gen:
            # Move the batch to the same device as the model
            device = next(net.parameters()).device
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()  # reset gradients

            out, _ = net(x)  # forward pass

            loss = criterion(out, y)  # compute loss

            loss.backward()  # backprop through SNN

            optimizer.step()  # apply Adam update

            with torch.no_grad():
                for layer in net.layers:
                    layer.tau.data.clamp_(
                        min=0.001
                    )  # stabilise tau values

        # Validation per epoch
        val_loss, _ = evaluate_loss(
            net,
            val_data,
        )

        hist.append(val_loss)

    return net, hist