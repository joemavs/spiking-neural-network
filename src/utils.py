import random
from pathlib import Path

import numpy as np
import torch


def get_device():
    """
    Use a CUDA GPU when available, otherwise use the CPU.
    """

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def set_random_seed(seed=42):
    """
    Set random seeds to make results more reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set the seed for all available CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clamp_time_constants(model, minimum=0.001):
    """
    Clamp trainable time constants to ensure they stay positive.

    A zero or negative tau would make the neuron dynamics invalid.
    """

    with torch.no_grad():

        for layer in model.layers:

            # Only modify layers that contain a tau parameter
            if hasattr(layer, "tau"):
                layer.tau.clamp_(min=minimum)


def save_checkpoint(
    path,
    model,
    optimizer=None,
    epoch=None,
    history=None,
):
    """
    Save the model and optional training information.

    Unlike saving only model.state_dict(), a checkpoint can also
    preserve optimiser state, epoch number and loss history.
    """

    path = Path(path)

    # Create the parent directory if it does not already exist
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = (
            optimizer.state_dict()
        )

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if history is not None:
        checkpoint["history"] = history

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
):
    """
    Load a previously saved model checkpoint.

    Returns:
        checkpoint : dict
            Complete checkpoint information, including any saved
            epoch number or training history.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    # Restore model parameters
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Restore optimiser state when an optimiser was supplied
    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint


def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a model.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def simple_moving_average(data, window_size):
    """
    Compute a simple moving average.

    This is useful for smoothing velocity or loss curves for plotting.
    """

    data = np.asarray(data)

    if window_size <= 0:
        raise ValueError(
            "window_size must be greater than zero."
        )

    if window_size > len(data):
        raise ValueError(
            "window_size cannot be greater than the data length."
        )

    weights = np.ones(window_size) / window_size

    # Return an output with the same length as the input
    moving_average = np.convolve(
        data,
        weights,
        mode="same",
    )

    return moving_average