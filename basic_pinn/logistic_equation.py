from typing import Callable
import argparse

import torch
from torch import nn
import numpy as np

from pinn import make_forward_fn, LinearNN
from plotting import plot_1d_solution
from training import train_pinn


R = 1.0  # rate of maximum population growth parameterizing the equation
X_BOUNDARY = 0.0  # boundary condition coordinate
F_BOUNDARY = 0.5  # boundary condition value


def make_loss_fn(f: Callable, dfdx: Callable) -> Callable:
    """Make a function loss evaluation function

    The loss is computed as sum of the interior MSE loss (the differential equation residual)
    and the MSE of the loss at the boundary

    Args:
        f (Callable): The functional forward pass of the model used a universal function approximator. This
            is a function with signature (x, params) where `x` is the input data and `params` the model
            parameters
        dfdx (Callable): The functional gradient calculation of the universal function approximator. This
            is a function with signature (x, params) where `x` is the input data and `params` the model
            parameters

    Returns:
        Callable: The loss function with signature (params, x) where `x` is the input data and `params` the model
            parameters. Notice that a simple call to `dloss = functorch.grad(loss_fn)` would give the gradient
            of the loss with respect to the model parameters needed by the optimizers
    """

    def loss_fn(params: torch.Tensor, x: torch.Tensor):

        # interior loss
        f_value = f(x, params)
        interior = dfdx(x, params) - R * f_value * (1 - f_value)

        # boundary loss
        x0 = X_BOUNDARY
        f0 = F_BOUNDARY
        x_boundary = torch.tensor([x0])
        f_boundary = torch.tensor([f0])
        boundary = f(x_boundary, params) - f_boundary

        loss = nn.MSELoss()
        loss_value = loss(interior, torch.zeros_like(interior)) + loss(
            boundary, torch.zeros_like(boundary)
        )

        return loss_value

    return loss_fn


if __name__ == "__main__":

    # make it reproducible
    torch.manual_seed(42)

    # parse input from user
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-hidden", type=int, default=5)
    parser.add_argument("-d", "--dim-hidden", type=int, default=5)
    parser.add_argument("-b", "--batch-size", type=int, default=30)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-1)
    parser.add_argument("-e", "--num-epochs", type=int, default=100)

    args = parser.parse_args()

    # configuration
    num_hidden: int = args.num_hidden
    dim_hidden: int = args.dim_hidden
    batch_size: int = args.batch_size
    num_iter: int = args.num_epochs
    tolerance = 1e-8
    learning_rate: float = args.learning_rate
    
    domain = (-5.0, 5.0)

    # function versions of model forward, gradient and loss
    model = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=1)
    funcs = make_forward_fn(model, derivative_order=1)

    f = funcs[0]
    dfdx = funcs[1]
    loss_fn = make_loss_fn(f, dfdx)

    def domain_sampler() -> torch.Tensor:
        return torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])

    opt_params, loss_evolution = train_pinn(
        model, 
        loss_fn, 
        domain_sampler, 
        learning_rate=learning_rate, 
        num_iter=num_iter,
    )

    x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)
    x_sample = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])
    f_eval = f(x_eval, tuple(opt_params))

    def analytical_sol_fn(x):
        return 1.0 / (1.0 + (1.0 / F_BOUNDARY - 1.0) * np.exp(-R * x))
    
    plot_1d_solution(x_eval, x_sample, f_eval, analytical_sol_fn, loss_evolution, show=True)
