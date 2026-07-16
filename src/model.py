import torch
from torch import nn


class SurrogateHeaviside(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrogateHeaviside.scale*torch.abs(input)+1.0)**2
        return grad

# here we overwrite our naive spike function by the "SurrogateHeaviside" nonlinearity which implements a surrogate gradient
surrogate_heaviside  = SurrogateHeaviside.apply


class SNNLayer(nn.Module):
    def __init__(self, n_in, n_out, spiking=True, dt=1e-3):
        """
        n_in: number of input neurons
        n_out: number of output neurons
        spiking: whether this layer is spiking (True) or non-spiking (False)
        dt: simulation time step (s)
        """
        super(SNNLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.spiking = spiking
        self.dt = dt

        # Initialise trainable weight matrix
        self.w = nn.Parameter(0.15* torch.randn(n_in, n_out))

        # Initialize time constants differently for spiking vs non-spiking neurons
        if spiking:
            tau_min, tau_max = 20*dt, 100*dt # 20–100 ms spread for LIF neurons
        else:
            tau_min, tau_max = 200*dt, 1000*dt # longer time constants

        # Voltage parameters
        self.v_rest = 0.0      # resting potential
        self.v_reset = 0.0     # reset potential after spike
        self.v_th = 1.0        # spike threshold

        # Initialize trainable tau in the earlier specified range
        tau_init = torch.rand(n_out) * (tau_max - tau_min) + tau_min
        self.tau = nn.Parameter(tau_init)

    def forward(self, x):
            """
            Forward pass of the LIF layer.

            Args:
              x: tensor of shape (batch_size, n_in, num_time_points)
                Input spike trains (0 or 1)

            Returns:
              y: tensor of shape (batch_size, n_out, num_time_points)
                Spiking output (0/1) if spiking=True, or membrane potential otherwise
              v_trace: tensor of shape (batch_size, n_out, num_time_points)
                      Tracks membrane potential over time for all neurons
            """

            B, n_in, T = x.shape # batch size, input neurons, time points extracted from shape of x

            # Compute input current for each neuron: I(t) = w_i * x_i(t)
            # x: (B, n_in, T) -> transpose to (B, T, n_in) to match matmul with w
            # w: (n_in, n_out)
            # result: (B, T, n_out), then transposed back to (B, n_out, T)
            I = torch.matmul(x.transpose(1, 2), self.w)
            I = I.transpose(1, 2)

            # Initialise membrane potential and output trace
            v = torch.zeros((B, self.n_out), device=x.device) # current voltage
            v_trace = torch.zeros((B, self.n_out, T), device=x.device) # recorded voltage over time
            y = torch.zeros((B, self.n_out, T), device=x.device) # output (spikes or voltage)

            # Reshape tau to broadcast over batch
            tau = self.tau.view(1, self.n_out)
            # Precompute alpha = exp(-dt / tau) for leaky integration
            alpha = torch.exp(-self.dt / tau)

            # Time-stepping loop
            for t in range(T):
                # Update membrane potential: v(t) = alpha * v(t-1) + I(t)
                v = alpha*v+I[:, :, t]
                v_trace[:, :, t] = v # record voltage

                if self.spiking:
                    # Generate spikes using surrogate Heaviside function
                    spikes = surrogate_heaviside(v - self.v_th)  # threshold at 1.0
                    # Reset voltage after spike
                    v = v * (1.0 - spikes) + self.v_reset * spikes # reset voltage to v_reset where spikes occur, keep others unchanged
                    y[:, :, t] = spikes
                else:
                    # Non-spiking mode, just records voltage
                    y[:, :, t] = v

            return y, v_trace


class SNNNetwork(nn.Module):
    def __init__(self, layers):
        """
        A spiking neural network container.

        Args:
            layers (list of nn.Module): List of layers (e.g., SNNLayer instances) that form the network.
        """

        super().__init__()
        # Store the layers as a ModuleList so PyTorch tracks their parameters
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Store outputs from all layers
        layer_outputs = []
        input_signal = x  # start with input

        # Pass the input through each layer in sequence
        for layer in self.layers:
            out, _ = layer(input_signal)  # Compute the output of the current layer
            layer_outputs.append(out)   # Save this layer's output
            # Output of this layer becomes input to the next layer
            input_signal = out

        return out, layer_outputs