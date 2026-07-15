import torch
from torch import nn


class SurrogateHeaviside(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we are able to use PyTorch's autograd functionality.

    Here we use the normalized negative part of a fast sigmoid,
    as this was done in Zenke & Ganguli (2018).
    """

    # Controls the steepness of the surrogate gradient
    scale = 100.0

    @staticmethod
    def forward(ctx, input_tensor):
        """
        In the forward pass, compute a step function of the input tensor
        and return it.

        ctx is a context object used to store information needed later
        when backpropagating the error signals. To achieve this, we use
        the ctx.save_for_backward method.
        """

        # Save the input so it can be used during the backward pass
        ctx.save_for_backward(input_tensor)

        # Create an output tensor with the same shape as the input
        out = torch.zeros_like(input_tensor)

        # Values above zero produce a spike
        out[input_tensor > 0] = 1.0

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, receive the gradient of the loss with
        respect to the output and calculate a surrogate gradient with
        respect to the input.

        Here we use the normalized negative part of a fast sigmoid,
        as this was done in Zenke & Ganguli (2018).
        """

        # Retrieve the input saved during the forward pass
        (input_tensor,) = ctx.saved_tensors

        # Clone the incoming gradient
        grad_input = grad_output.clone()

        # Calculate the surrogate gradient
        grad = grad_input / (
            SurrogateHeaviside.scale * torch.abs(input_tensor) + 1.0
        ) ** 2

        return grad


# Here we overwrite our naive spike function with the
# SurrogateHeaviside nonlinearity, which implements a surrogate gradient
surrogate_heaviside = SurrogateHeaviside.apply


class SNNLayer(nn.Module):
    """
    A layer of leaky integrate-and-fire neurons.

    The layer can operate in either:

    - spiking mode, where the outputs are binary spikes
    - non-spiking mode, where the outputs are membrane potentials
    """

    def __init__(self, n_in, n_out, spiking=True, dt=1e-3):
        """
        Args:
            n_in : int
                Number of input neurons.
            n_out : int
                Number of output neurons.
            spiking : bool
                Whether this layer is spiking (True) or
                non-spiking (False).
            dt : float
                Simulation time step in seconds.
        """

        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.spiking = spiking
        self.dt = dt

        # Initialise trainable weight matrix
        self.w = nn.Parameter(
            0.15 * torch.randn(n_in, n_out)
        )

        # Initialise time constants differently for spiking
        # and non-spiking neurons
        if spiking:
            # 20–100 ms spread for LIF neurons
            tau_min = 20 * dt
            tau_max = 100 * dt
        else:
            # Longer time constants for non-spiking output neurons
            tau_min = 200 * dt
            tau_max = 1000 * dt

        # Voltage parameters
        self.v_rest = 0.0   # resting potential
        self.v_reset = 0.0  # reset potential after spike
        self.v_th = 1.0     # spike threshold

        # Initialise trainable tau in the previously specified range
        tau_init = (
            torch.rand(n_out) * (tau_max - tau_min) + tau_min
        )

        self.tau = nn.Parameter(tau_init)

    def forward(self, x):
        """
        Forward pass of the LIF layer.

        Args:
            x : torch.Tensor
                Tensor of shape
                (batch_size, n_in, num_time_points).

                Contains the input spike trains or the outputs from
                the preceding layer.

        Returns:
            y : torch.Tensor
                Tensor of shape
                (batch_size, n_out, num_time_points).

                Contains spiking output if spiking=True, or membrane
                potential output if spiking=False.

            v_trace : torch.Tensor
                Tensor of shape
                (batch_size, n_out, num_time_points).

                Tracks membrane potential over time for all neurons.
        """

        # Extract batch size, input neurons and time points
        # from the shape of x
        batch_size, n_in, num_time_points = x.shape

        if n_in != self.n_in:
            raise ValueError(
                f"Expected {self.n_in} input neurons, "
                f"but received {n_in}."
            )

        # Compute input current for each neuron:
        # I(t) = w_i * x_i(t)
        #
        # x: (batch_size, n_in, num_time_points)
        # transpose to: (batch_size, num_time_points, n_in)
        #
        # w: (n_in, n_out)
        #
        # result: (batch_size, num_time_points, n_out)
        # then transpose back to:
        # (batch_size, n_out, num_time_points)
        input_current = torch.matmul(
            x.transpose(1, 2),
            self.w,
        )

        input_current = input_current.transpose(1, 2)

        # Initialise membrane potential
        voltage = torch.full(
            (batch_size, self.n_out),
            self.v_rest,
            device=x.device,
            dtype=x.dtype,
        )

        # Recorded voltage over time
        voltage_trace = torch.zeros(
            (batch_size, self.n_out, num_time_points),
            device=x.device,
            dtype=x.dtype,
        )

        # Output spikes or voltage
        output = torch.zeros(
            (batch_size, self.n_out, num_time_points),
            device=x.device,
            dtype=x.dtype,
        )

        # Reshape tau so that it broadcasts over the batch dimension
        tau = self.tau.view(1, self.n_out)

        # Keep the time constants positive and prevent division by zero
        tau = torch.clamp(tau, min=self.dt)

        # Precompute alpha = exp(-dt / tau) for leaky integration
        alpha = torch.exp(-self.dt / tau)

        # Time-stepping loop
        for time_index in range(num_time_points):

            # Update membrane potential:
            # v(t) = alpha * v(t-1) + I(t)
            voltage = (
                alpha * voltage
                + input_current[:, :, time_index]
            )

            if self.spiking:
                # Generate spikes using the surrogate
                # Heaviside function
                spikes = surrogate_heaviside(
                    voltage - self.v_th
                )

                # Record the voltage before resetting it
                voltage_trace[:, :, time_index] = voltage

                # Reset voltage after a spike
                #
                # Reset voltage to v_reset where spikes occur,
                # while keeping other voltages unchanged
                voltage = (
                    voltage * (1.0 - spikes)
                    + self.v_reset * spikes
                )

                output[:, :, time_index] = spikes

            else:
                # Non-spiking mode simply records voltage
                voltage_trace[:, :, time_index] = voltage
                output[:, :, time_index] = voltage

        return output, voltage_trace


class SNNNetwork(nn.Module):
    """
    A spiking neural network container.

    The network passes its input through a sequence of SNNLayer
    instances.
    """

    def __init__(self, layers):
        """
        Args:
            layers : list of nn.Module
                List of layers, such as SNNLayer instances, that form
                the network.
        """

        super().__init__()

        # Store the layers as a ModuleList so PyTorch tracks
        # their trainable parameters
        self.layers = nn.ModuleList(layers)

        if len(self.layers) == 0:
            raise ValueError(
                "SNNNetwork requires at least one layer."
            )

    def forward(self, x):
        """
        Pass the input through each layer in sequence.

        Args:
            x : torch.Tensor
                Input tensor with shape
                (batch_size, input_neurons, num_time_points).

        Returns:
            output : torch.Tensor
                Output of the final layer.

            layer_outputs : list of torch.Tensor
                Output from every layer in the network.

            voltage_traces : list of torch.Tensor
                Membrane-potential trace from every layer.
        """

        # Store outputs from all layers
        layer_outputs = []

        # Store voltage traces from all layers
        voltage_traces = []

        # Start with the network input
        input_signal = x

        # Pass the input through each layer in sequence
        for layer in self.layers:

            # Compute the output of the current layer
            output, voltage_trace = layer(input_signal)

            # Save this layer's output
            layer_outputs.append(output)

            # Save this layer's membrane-potential trace
            voltage_traces.append(voltage_trace)

            # Output of this layer becomes input to the next layer
            input_signal = output

        return output, layer_outputs, voltage_traces