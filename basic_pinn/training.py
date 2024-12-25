from typing import Callable

import torch
import torchopt
from torch import Tensor


def train_pinn(
    model: torch.nn.Module, 
    loss_fn: Callable,
    domain_sampler: Callable,
    learning_rate: float = 1e-3,
    num_iter: int = 100,
) -> tuple[tuple[Tensor, ...], Tensor]:

    print_every = num_iter // min(num_iter, 50)

    # choose optimizer with functional API using functorch
    optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))

    # initial parameters, randomly initialized
    params = tuple(model.parameters())

    # train the model
    loss_evolution = []
    for i in range(num_iter):

        # sample points in the domain for each epoch
        x = domain_sampler()

        # compute the loss with the current parameters
        loss = loss_fn(params, x)

        # update the parameters with functional optimizer
        params = optimizer.step(loss, params)

        if i % print_every == 0:
            print(f"Iteration {i} with loss {float(loss)}")
        
        loss_evolution.append(float(loss))

    return params, loss_evolution