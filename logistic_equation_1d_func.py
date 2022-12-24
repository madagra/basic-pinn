from typing import Callable
import argparse

import matplotlib.pyplot as plt
import torch
from functorch import make_functional, grad, vmap
from torch import nn
import numpy as np
import torchopt


X_BOUNDARY = 0.0
F_BOUNDARY = 0.5


class NNApproximator(nn.Module):
    def __init__(
        self, 
        num_inputs: int = 1,
        num_outputs: int = 1,
        num_hidden: int = 1, 
        dim_hidden: int = 1, 
        act: nn.Module = nn.Tanh()
    ) -> None:
        """Simple neural network with linear layers and non-linear activation function
        
        This class is used as universal function approximator for the solution of
        partial differential equations using PINNs

        Args:
            num_inputs (int, optional): The number of input dimensions
            num_outputs (int, optional): The number of outputs of the model, in general is 1
            num_hidden (int, optional): The number of hidden layers in the mode
            dim_hidden (int, optional): The number of neurons for each hidden layer
            act (nn.Module, optional): The type of non-linear activation function to be used
        """
        super().__init__()

        self.layer_in = nn.Linear(num_inputs, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, num_outputs)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.layer_in(x))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        return self.layer_out(out)


def make_forward_fn(
    num_hidden: int = 1, dim_hidden: int = 1
) -> tuple[Callable, tuple[torch.Tensor], Callable, Callable]:
    """Make a functional forward pass and gradient functions

    This function creates a functional model using the NNApproximator class
    and then returns a tuple with the functional model itself, the parameters
    initialized randomly and the composable v-mapped version of the forward pass
    and gradient function for efficient batching over the inputs
    
    The forward pass is created as a closure
    
    Args:
        num_hidden (int, optional): Number of hidden layers in the NN model. Defaults to 1.
        dim_hidden (int, optional): Number of neurons per hidden layer in the NN model. Defaults to 1.

    Returns:
        tuple: A tuple with the functional model itself, the parameters
            and the composable v-mapped version of the forward pass and
            gradient function for efficient batching over the inputs
    """
    model = NNApproximator(num_hidden=num_hidden, dim_hidden=dim_hidden)    
    fmodel, params = make_functional(model)
    
    def f(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # only a single element is support thus unsqueeze must be applied
        # for batching, vmap must be used as below
        x_ = x.unsqueeze(0)
        res = fmodel(params, x_).squeeze(0)
        return res

    f_vmap = vmap(f, in_dims=(0, None))
    dfdx_vmap = vmap(grad(f), in_dims=(0, None))
    
    return fmodel, params, f_vmap, dfdx_vmap


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
        interior = dfdx(x, params) - f_value * (1 - f_value)
        
        # boundary loss
        x0 = X_BOUNDARY
        f0 = F_BOUNDARY
        x_boundary = torch.tensor([x0])
        f_boundary = torch.tensor([f0])
        boundary = f(x_boundary, params) - f_boundary

        loss = nn.MSELoss()
        loss_value = loss(interior, torch.zeros_like(interior)) + loss(boundary, torch.zeros_like(boundary))

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
    parser.add_argument("-lr","--learning-rate", type=float, default=1e-1)

    args = parser.parse_args()
    
    # configuration
    num_hidden = args.num_hidden
    dim_hidden = args.dim_hidden
    batch_size = args.batch_size
    num_iter = 100
    tolerance = 1e-8
    learning_rate = args.learning_rate
    domain = (-5.0, 5.0)
        
    # function versions of model forward, gradient and loss
    fmodel, params, f, dfdx = make_forward_fn(num_hidden=num_hidden, dim_hidden=dim_hidden)
    loss_fn = make_loss_fn(f, dfdx)

    # choose optimizer with functional API using functorch
    optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))

    # train the model
    for i in range(num_iter):
        
        # sample points in the domain randomly for each epoch
        x = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])

        # update the parameters
        loss = loss_fn(params, x)
        params = optimizer.step(loss, params)
        
        print(f"Iteration {i} with loss {float(loss)}")
    
    # plot solution on the given domain
    x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)
    f_eval = f(x_eval, params)
    analytical_sol_fn = lambda x: 1.0 / (1.0 + np.exp(-1.0 * x))
    x_eval_np = x_eval.detach().numpy()
    
    fig, ax = plt.subplots()

    ax.plot(x_eval_np, f_eval.detach().numpy(), label="PINN final solution")
    ax.plot(
        x_eval_np,
        analytical_sol_fn(x_eval_np),
        label=f"Analytic solution",
        color="green",
        alpha=0.75,
    )
    ax.set(title="Logistic equation solved with NNs", xlabel="t", ylabel="f(t)")
    ax.legend()
    
    plt.show()
