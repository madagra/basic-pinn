from typing import Callable
import argparse

import torch
from torch import nn
import matplotlib.pyplot as plt

from pinn import make_forward_fn_nd, LinearNN, Config
from training import train_pinn
from plotting import animate_2d_solution

C = 1.0  # wave speed

# initial and final positions where the string is attached
X_I = 0.
X_F = torch.pi

# boundary conditions for the displacement
F_0 = 0.  # displacement at the beginning of the string
F_L = 0.  # displacement at the end of the string

# boundary conditions for the elapsed time
T_I = 0.  # initial time
T_F = 4.  # final time

 
def icond_u(x: torch.Tensor) -> torch.Tensor:
    """Initial condition of the wave equation displacement

    Mathematical formulation: u(x, t = 0) = sin(x)
    """
    return torch.sin(x)


def icond_ut(x: torch.Tensor) -> torch.Tensor:
    """Initial condition of the wave equation velocity

    Mathematical formulation: du/dt(x, t = 0) = 0
    """
    return torch.zeros_like(x)


def make_loss_fn(
    f: Callable,
    dfdt: Callable,
    d2fdx2: Callable,
    d2fdt2: Callable
) -> Callable:
    """Make a function loss evaluation function

    The loss is computed as sum of the interior MSE loss (the differential equation residual)
    and the MSE of the loss at the boundary

    Args:
        f (Callable): The functional forward pass of the model used a universal function approximator. This
            is a function with signature (x, params) where `x` is the input data and `params` the model
            parameters
        dfdt (Callable): First order pure partial derivative w.r.t. t. This is computed via the functional
            `grad` higher-order function and it supports batches
        d2fdx2 (Callable): Second order pure partial derivative w.r.t. x. This is computed via repeated application
            of the functional `grad` higher-order function and it supports batches
        d2fdt2 (Callable): Second order pure partial derivative w.r.t. t. This is computed via repeated application
            of the functional `grad` higher-order function and it supports batches

    Returns:
        Callable: The loss function with signature (params, x) where `x` is the input data and `params` the model
            parameters. Notice that a simple call to `dloss = functorch.grad(loss_fn)` would give the gradient
            of the loss with respect to the model parameters needed by the optimizers
    """

    def loss_fn(params: torch.Tensor, input: torch.Tensor | tuple[torch.Tensor, ...]):
        
        t, x = input

        # interior loss
        interior = d2fdt2(x, t, params=params) - (C ** 2) * d2fdx2(x, t, params=params)

        # boundary conditions
        x0_val = torch.ones_like(x) * X_I
        f0_val = torch.ones_like(x) * F_0
        boundary_x0 = f(x0_val, t, params=params) - f0_val

        xL_val = torch.ones_like(x) * X_F
        fL_val = torch.ones_like(x) * F_L
        boundary_xL = f(xL_val, t, params=params) - fL_val

        # initial condition on displacement
        t_initial = torch.ones_like(t) * T_I
        f_initial = icond_u(x)
        f_initial_val = f(x, t_initial, params=params) - f_initial

        # initial condition on velocity
        dfdt_initial = icond_ut(x)
        dfdt_initial_val = dfdt(x, t_initial, params=params) - dfdt_initial

        loss = nn.MSELoss()
        loss_value = \
            loss(interior, torch.zeros_like(interior)) + \
            loss(boundary_x0, torch.zeros_like(boundary_x0)) + \
            loss(boundary_xL, torch.zeros_like(boundary_xL)) + \
            loss(f_initial_val, torch.zeros_like(f_initial_val)) + \
            loss(dfdt_initial, torch.zeros_like(dfdt_initial_val))

        return loss_value

    return loss_fn


if __name__ == "__main__":

    # make it reproducible
    torch.manual_seed(42)

    # parse input from user
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-hidden", type=int, default=5)
    parser.add_argument("-d", "--dim-hidden", type=int, default=5)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-1)
    parser.add_argument("-e", "--num-epochs", type=int, default=10_000)

    args = parser.parse_args()
    config = Config(**args.__dict__)

    domain_x = (X_I, X_F)
    domain_t = (T_I, T_F)

    def domain_sampler() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.FloatTensor(config.batch_size).uniform_(domain_x[0], domain_x[1])
        t, _ = torch.sort(torch.FloatTensor(config.batch_size).uniform_(domain_t[0], domain_t[1]))
        t_and_x = torch.cartesian_prod(t, x)
        return t_and_x[:, 0], t_and_x[:, 1]

    # MLP model
    model = LinearNN(num_layers=config.num_hidden, num_neurons=config.dim_hidden, num_inputs=2)

    f, dfdt, d2fdt2 = make_forward_fn_nd(model, on_variable=0, derivative_order=2)
    _, _, d2fdx2 = make_forward_fn_nd(model, on_variable=1, derivative_order=2)

    loss_fn = make_loss_fn(f, dfdt, d2fdx2, d2fdt2)

    inputs = domain_sampler()
    initial_params = tuple(model.parameters())
    initial_loss = loss_fn(initial_params, inputs)
    print(f"Initial loss: {initial_loss.item()}")

    opt_params, loss_evolution = train_pinn(
        model, 
        loss_fn, 
        domain_sampler, 
        learning_rate=config.learning_rate, 
        num_iter=config.num_epochs,
    )

    x_eval = torch.arange(domain_x[0], domain_x[1], 0.01)
    t_eval = torch.arange(domain_t[0], domain_t[1], 0.1)

    _, ani = animate_2d_solution(x_eval, t_eval, opt_params, f, show=True)

    ani.save("wave_equation_1d.gif", writer="pillow")
