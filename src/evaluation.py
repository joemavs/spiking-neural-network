import torch
from torch import nn


def compute_loss(net, data_generator, null_model=False):
    """
    Compute the mean squared error (MSE) over a dataset.

    Args:
        net (nn.Module): The spiking neural network to evaluate.
        data_generator (iterable): Generator that yields batches of (x, y) data.
        null_model (bool, optional): If True, computes loss assuming
            the network predicts all zeros.

    Returns:
        float: Mean MSE over all batches.
    """

    # Define the loss function (mean squared error)
    loss_fn = nn.MSELoss()

    # Initialize counts for total loss and number of batches
    total_loss = 0.0
    count = 0

    # Set network to evaluation mode
    net.eval()

    # Find which device the model is stored on
    device = next(net.parameters()).device

    with torch.no_grad():  # Disable gradients for speed/memory
        for x, y in data_generator:  # Iterate over batches from generator

            # Move data to the same device as the model
            x, y = x.to(device), y.to(device)

            if null_model:
                # Null model: predict zeros for all outputs
                pred = torch.zeros_like(y)
            else:
                # SNN Model: Get prediction from network
                pred, _ = net(x)

            # Compute MSE for this batch
            loss = loss_fn(pred, y)

            # Accumulate total loss
            total_loss += loss.item()

            # Count this batch
            count += 1

    if count == 0:
        raise ValueError("The data generator produced no batches.")
    
    # Return average loss over all batches
    return total_loss / count



def evaluate_loss(net, data_generator):
    """
    Evaluates the model against a null baseline.

    Args:
        net : nn.Module
            Network to evaluate.

        data_generator : callable
            Function that returns a fresh generator of batches.

    Returns:
        model_loss : float
            Mean squared error of the network.

        null_model_loss : float
            Mean squared error of the null model.
    """

    # Batch generator for validation or test data
    gen = data_generator()
    model_loss = compute_loss(net, gen)  # Compute loss

    # A new generator is needed because the first generator
    # has already been consumed
    gen_null = data_generator()
    null_model_loss = compute_loss(
        net,
        gen_null,
        null_model=True,
    )  # Compute null loss

    return model_loss, null_model_loss