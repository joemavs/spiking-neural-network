import os
from functools import partial

import torch

from src.dataset import (
    batched_data,
    create_data_splits,
    load_data,
    whiten_velocity,
)
from src.model import SNNLayer, SNNNetwork
from src.training import train_model


# Use GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Data and model paths
data_file = "data/s1_data_raw.mat"
model_file = "checkpoints/model_spiking.pth"


# Fixed config for final training
dt = 1e-3
n_output = 2
length = 1.0
num_epochs = 10
range_to_use = 1


# Best hyperparameters found during tuning with Optuna
lr = 0.0072755292043824105
n_hidden = 278
batch_size = 32
weight_decay = 6.461889973562118e-05
init_scale = 0.08083323891499462


def main():
    """
    Loads the dataset, builds the spiking model,
    trains it and saves the trained parameters.
    """

    print(f"Using device: {device}")

    # Load the raw data
    spike_times, vel, vel_times = load_data(
        data_file
    )

    # Whiten the velocity data
    velw = whiten_velocity(vel)

    # Create training, validation and test splits
    train_times, val_times, test_times = (
        create_data_splits(
            vel_times,
            segment_length=length,
        )
    )

    # Number of input neurons
    n_input = len(spike_times)

    # Build network
    l1 = SNNLayer(
        n_input,
        n_hidden,
        spiking=True,
        dt=dt,
    )  # hidden spiking layer

    l2 = SNNLayer(
        n_hidden,
        n_output,
        spiking=False,
        dt=dt,
    )  # non-spiking readout layer

    net = SNNNetwork(
        [l1, l2]
    ).to(device)

    # Weight initialisation
    with torch.no_grad():

        # Initialise input weights
        l1.w.data = (
            init_scale
            * torch.randn(
                n_input,
                n_hidden,
                device=device,
            )
        )

        # Initialise output weights
        l2.w.data = (
            0.01
            * torch.randn(
                n_hidden,
                n_output,
                device=device,
            )
        )

    # Optimiser
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Store the dataset arguments shared by all generators
    data_generator = partial(
        batched_data,
        spike_times=spike_times,
        vel_times=vel_times,
        velw=velw,
        train_times=train_times,
        val_times=val_times,
        test_times=test_times,
        range_to_use=range_to_use,
        dt=dt,
        length=length,
        batch_size=batch_size,
    )

    # A new generator is created every time train_data()
    # or val_data() is called
    train_data = partial(
        data_generator,
        data_split="train",
    )

    val_data = partial(
        data_generator,
        data_split="val",
    )

    print("Training spiking model...")

    # Train model
    net, history = train_model(
        net=net,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        num_epochs=num_epochs,
        label="Spiking",
    )

    # Create checkpoint folder if it does not exist
    os.makedirs(
        os.path.dirname(model_file),
        exist_ok=True,
    )

    # Save final model
    torch.save(
        net.state_dict(),
        model_file,
    )

    print(f"Saved spiking model to {model_file}")

    if history:
        print(
            "Final validation loss: "
            f"{history[-1]:.5f}"
        )


if __name__ == "__main__":
    main()