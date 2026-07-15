import numpy as np
import torch
from torch import nn

from .evaluation import evaluate_network
from .utils import clamp_time_constants, save_checkpoint


def train_model(
    model,
    train_data,
    val_data,
    num_epochs=10,
    learning_rate=1e-3,
    weight_decay=0.0,
    max_num_batches=None,
    device="cpu",
    checkpoint_path=None,
):
    """
    Train the network to minimise the mean squared error between
    the output of the network and the recorded velocities.

    Args:
        model : torch.nn.Module
            Network to train.

        train_data : callable
            Function that returns a fresh generator of training batches.

            Each batch should contain:

                x : torch.Tensor
                    Shape
                    (batch_size, num_neurons, num_time_points).

                y : torch.Tensor
                    Shape
                    (batch_size, 2, num_time_points).

        val_data : callable
            Function that returns a fresh generator of validation batches.

        num_epochs : int
            Number of complete passes over the training data.

        learning_rate : float
            Learning rate for the optimiser.

        weight_decay : float
            L2 regularisation used by the optimiser.

        max_num_batches : int or None
            Maximum number of training batches used per epoch.
            If None, every available batch is used.

        device : str or torch.device
            Device on which the model should be trained.

        checkpoint_path : str or None
            Optional path used to save the best model.

    Returns:
        history : dict
            Training and validation loss for every epoch.
    """

    # Move the network to the selected device
    model = model.to(device)

    # Loss function: Mean Squared Error
    criterion = nn.MSELoss()

    # Optimizer: Adam
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Stores training and validation loss per epoch
    history = {
        "train_loss": [],
        "val_loss": [],
        "null_val_loss": [],
    }

    # Store the best validation loss so the best model can be saved
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(num_epochs):

        # Set the model to training mode
        model.train()

        # Track loss for this epoch
        local_loss = []

        # A fresh generator must be created at the start of every epoch
        for batch_index, (x, y) in enumerate(train_data()):

            # Stop early when only a limited number of batches are required
            if (
                max_num_batches is not None
                and batch_index >= max_num_batches
            ):
                break

            # Move data to the selected device
            x = x.to(device)
            y = y.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            y_out, _, _ = model(x)

            # Compute loss
            loss = criterion(y_out, y)

            # Backpropagation
            loss.backward()

            # Update weights and time constants
            optimizer.step()

            # Clamp tau parameters to ensure they stay positive
            clamp_time_constants(model, minimum=0.001)

            # Record batch loss
            local_loss.append(loss.item())

        if not local_loss:
            raise RuntimeError(
                "The training data generator produced no batches. "
                "Check the batch size and number of available segments."
            )

        # Compute average training loss for the epoch
        train_loss = float(np.mean(local_loss))

        # Validation
        validation_results = evaluate_network(
            model=model,
            data_generator=val_data,
            device=device,
        )

        val_loss = validation_results["loss"]
        null_val_loss = validation_results["null_loss"]

        # Record losses
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["null_val_loss"].append(null_val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"train loss: {train_loss:.6f} | "
            f"validation loss: {val_loss:.6f} | "
            f"null loss: {null_val_loss:.6f}"
        )

        # Save the model when validation performance improves
        if checkpoint_path is not None and val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
            )

    return history