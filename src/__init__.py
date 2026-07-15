"""
Spiking neural network decoder package.

This package contains data-processing, model-training and evaluation
utilities for decoding movement velocity from neural spike recordings.
"""

from .dataset import (
    batched_data,
    create_splits,
    load_data,
    whiten_velocity,
)

from .evaluation import (
    calculate_component_mse,
    calculate_mse,
    compute_loss,
    evaluate_network,
    predict_batches,
)

from .model import (
    SNNLayer,
    SNNNetwork,
    SurrogateHeaviside,
    surrogate_heaviside,
)

from .training import train_model

from .utils import (
    clamp_time_constants,
    count_trainable_parameters,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_random_seed,
    simple_moving_average,
)

__all__ = [
    "SNNLayer",
    "SNNNetwork",
    "SurrogateHeaviside",
    "surrogate_heaviside",
    "load_data",
    "whiten_velocity",
    "create_splits",
    "batched_data",
    "train_model",
    "compute_loss",
    "evaluate_network",
    "predict_batches",
    "calculate_mse",
    "calculate_component_mse",
    "get_device",
    "set_random_seed",
    "clamp_time_constants",
    "save_checkpoint",
    "load_checkpoint",
    "count_trainable_parameters",
    "simple_moving_average",
]