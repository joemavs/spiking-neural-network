from scipy import io
import numpy as np
import torch


def load_data(data_path):
    """
    Load the raw neural recording data from a MATLAB file.

    Returns:
        spike_times : list of arrays
            Spike times (seconds) for each neuron.
        vel : ndarray
            Velocity data of shape (num_time_points, 2).
        vel_times : ndarray
            Times at which the velocities were recorded.
    """

    # Load the raw data
    data = io.loadmat(data_path)  # a MATLAB file!

    # A list of arrays of spike times in seconds, one for each neuron
    spike_times = [st[:, 0] for st in data["spike_times"].ravel()]

    # Velocity data shape (num_time_points, 2) for (x, y) coordinates
    vel = data["vels"]

    # Times the velocities were recorded
    vel_times = data["vel_times"].squeeze()

    return spike_times, vel, vel_times


def whiten_velocity(vel):
    """
    Whiten the velocity data by centering and scaling.
    """

    ## Whiten the velocity data

    # Whiten the velocity by subtracting the mean (centering)
    # and dividing by the standard deviation (scaling)

    # Make a copy so the original velocity array remains unchanged
    velw = vel.copy()

    # Subtract the mean
    velw -= np.mean(velw)

    # Divide by the standard deviation
    velw /= np.std(velw)

    return velw


def create_data_splits(vel_times, segment_length=1.0):
    """
    Create train, validation and test splits by dividing the recording
    into repeating 20-second blocks.
    """

    # We will split the data up into 1 second segments.
    # Every 20 seconds:
    # - the first 14 segments are allocated to training,
    # - the next 3 to validation,
    # - the final 3 to testing.
    # This ensures all three sets span the entire recording.

    t_min = vel_times[0]
    t_max = vel_times[-1]

    total_duration = t_max - t_min
    num_segments = int(total_duration // segment_length)

    # Calculate the start time of each segment
    segment_indices = np.arange(num_segments)
    segment_start_times = t_min + segment_indices * segment_length

    pattern_size = 20
    train_k = 14
    val_k = 3
    test_k = 3

    train_segments = []
    val_segments = []
    test_segments = []

    # Compute pattern position for each segment
    for idx in segment_indices:

        r = idx % pattern_size

        if r < train_k:
            train_segments.append(idx)

        elif r < train_k + val_k:
            val_segments.append(idx)

        else:
            test_segments.append(idx)

    # Store times for each split
    train_times = segment_start_times[train_segments]
    val_times = segment_start_times[val_segments]
    test_times = segment_start_times[test_segments]

    return train_times, val_times, test_times


def batched_data(
    spike_times,
    vel_times,
    velw,
    train_times,
    val_times,
    test_times,
    dt=1e-3,
    length=1,
    batch_size=64,
    range_to_use=1.0,
    data_split="train",
    device="cpu",
):
    """
    Generator that yields batches of spike trains and corresponding velocities.
    """

    # Calculate number of time bins per window
    num_time_points = int(length / dt)

    # Calculate number of neurons
    num_neurons = len(spike_times)

    # Choose split to use
    if data_split == "train":
        times_shuffled = train_times.copy()

    elif data_split == "test":
        times_shuffled = test_times.copy()

    else:
        times_shuffled = val_times.copy()

    # Shuffle sections within split randomly
    np.random.shuffle(times_shuffled)

    # Determine how many segments to use
    num_segments = int(np.floor(range_to_use * len(times_shuffled)))

    # Keep only desired fraction
    times_shuffled = times_shuffled[:num_segments]

    # Compute the number of batches
    num_batches = num_segments // batch_size

    # Iterate over each batch
    for batch_idx in range(num_batches):

        # Preallocate arrays for this batch
        x = torch.zeros(
            (batch_size, num_neurons, num_time_points),
            device=device,
        )

        y = torch.zeros(
            (batch_size, 2, num_time_points),
            device=device,
        )

        # Fill each item in the batch
        for b in range(batch_size):

            idx = batch_idx * batch_size + b

            # Compute start and end times of the window
            t0 = times_shuffled[idx]
            t_end = t0 + length

            times = t0 + np.arange(num_time_points) * dt

            # Bin spikes for each neuron in this time window
            for n in range(num_neurons):

                spikes_n = spike_times[n]

                # Select spikes that fall within the window
                mask = (spikes_n >= t0) & (spikes_n < t_end)
                spikes_here = spikes_n[mask]

                # Convert spike times to indices for the bins
                inds = ((spikes_here - t0) / dt).astype(int)

                # Keep only indices within valid range
                inds = inds[(inds >= 0) & (inds < num_time_points)]

                # Mark spikes with 1 in the batch array
                x[b, n, inds] = 1

            # Interpolate velocity components onto the same time grid
            vx = np.interp(times, vel_times, velw[:, 0])
            vy = np.interp(times, vel_times, velw[:, 1])

            # Store velocities in batch array
            y[b, 0, :] = torch.as_tensor(vx, device=device, dtype=torch.float32)
            y[b, 1, :] = torch.as_tensor(vy, device=device, dtype=torch.float32)

        yield x, y