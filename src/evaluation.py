import numpy as np
import torch
from torch import nn


def compute_loss(
    model,
    data_generator,
    device="cpu",
    null_model=False,
):
    """
    Compute mean squared error over all batches from a data generator.

    Args:
        model : torch.nn.Module
            Trained neural network.

        data_generator : callable
            Function that returns a fresh batch generator.

        device : str or torch.device
            Device on which evaluation should run.

        null_model : bool
            If True, use a prediction of zero instead of the model
            output. Since the velocity data have been centred, zero
            represents the mean-velocity baseline.

    Returns:
        mean_loss : float
            Mean loss across all evaluated batches.
    """

    criterion = nn.MSELoss()

    # Store loss for each batch
    losses = []

    # Set model to evaluation mode
    model.eval()

    # Disable gradients for efficiency
    with torch.no_grad():

        for x, y in data_generator():

            # Move data to the selected device
            x = x.to(device)
            y = y.to(device)

            if null_model:
                # The null model always predicts zero velocity
                prediction = torch.zeros_like(y)

            else:
                # Forward pass through the trained network
                prediction, _, _ = model(x)

            # Compute loss
            loss = criterion(prediction, y)

            losses.append(loss.item())

    if not losses:
        raise RuntimeError(
            "The evaluation data generator produced no batches. "
            "Check the batch size and number of available segments."
        )

    return float(np.mean(losses))


def evaluate_network(
    model,
    data_generator,
    device="cpu",
):
    """
    Evaluate the model against a null baseline.

    Returns:
        results : dict
            Contains the model loss, null-model loss and improvement
            over the null model.
    """

    # Compute network loss
    model_loss = compute_loss(
        model=model,
        data_generator=data_generator,
        device=device,
        null_model=False,
    )

    # Compute null-model loss
    null_loss = compute_loss(
        model=model,
        data_generator=data_generator,
        device=device,
        null_model=True,
    )

    # Positive values indicate that the trained network performs
    # better than the null model
    improvement = null_loss - model_loss

    return {
        "loss": model_loss,
        "null_loss": null_loss,
        "improvement": improvement,
    }


def predict_batches(
    model,
    data_generator,
    device="cpu",
):
    """
    Generate predictions for all batches returned by a data generator.

    Returns:
        predictions : np.ndarray
            Predicted velocity sequences.

        targets : np.ndarray
            Recorded velocity sequences.
    """

    predictions = []
    targets = []

    # Set model to evaluation mode
    model.eval()

    # Disable gradients during inference
    with torch.no_grad():

        for x, y in data_generator():

            # Move input data to the selected device
            x = x.to(device)

            # Forward pass
            y_out, _, _ = model(x)

            # Store predictions and targets on the CPU
            predictions.append(y_out.cpu())
            targets.append(y.cpu())

    if not predictions:
        raise RuntimeError(
            "The data generator produced no batches."
        )

    # Join all batches together
    predictions = torch.cat(predictions, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()

    return predictions, targets


def calculate_mse(predictions, targets):
    """
    Compute mean squared error between predictions and targets.
    """

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have the same shape. "
            f"Received {predictions.shape} and {targets.shape}."
        )

    return float(np.mean((predictions - targets) ** 2))


def calculate_component_mse(predictions, targets):
    """
    Compute separate mean squared errors for x and y velocity.

    The expected array shape is:

        (num_samples, 2, num_time_points)
    """

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions and targets must have the same shape."
        )

    if predictions.ndim != 3 or predictions.shape[1] != 2:
        raise ValueError(
            "Expected arrays with shape "
            "(num_samples, 2, num_time_points)."
        )

    # Compute loss for each velocity component
    x_mse = np.mean(
        (predictions[:, 0, :] - targets[:, 0, :]) ** 2
    )

    y_mse = np.mean(
        (predictions[:, 1, :] - targets[:, 1, :]) ** 2
    )

    return {
        "x_mse": float(x_mse),
        "y_mse": float(y_mse),
    }