from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.func import functional_call, grad, vmap, jacrev


@dataclass
class Config:
    num_hidden: int = 5
    dim_hidden: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-1
    num_epochs: int = 100


class LinearNN(nn.Module):
    def __init__(
        self,
        num_inputs: int = 1,
        num_layers: int = 1,
        num_neurons: int = 5,
        act: nn.Module = nn.Tanh(),
    ) -> None:
        """Basic neural network architecture with linear layers
        
        Args:
            num_inputs (int, optional): the dimensionality of the input tensor
            num_layers (int, optional): the number of hidden layers
            num_neurons (int, optional): the number of neurons for each hidden layer
            act (nn.Module, optional): the non-linear activation function to use for stitching
                linear layers togeter
        """
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.num_layers = num_layers

        layers = []

        # input layer
        layers.append(nn.Linear(self.num_inputs, num_neurons))

        # hidden layers with linear layer and activation
        for _ in range(num_layers):
            layers.extend([nn.Linear(num_neurons, num_neurons), act])

        # output layer
        layers.append(nn.Linear(num_neurons, 1))

        # build the network
        self.network = nn.Sequential(*layers)

    def forward(self, *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        
        if self.num_inputs == 1:
            input = inputs[0]
            return self.network(input.reshape(-1, 1)).squeeze()
        else:

            # NOTE
            # when receiving vectors from the application of a grad
            # higher-order function, the type of the tensor is slightly
            # different and their shape is not defined. With these tensors, 
            # stacking works on dimension 0, thus the definition
            # of the `dim` variable below
            dim = int(all([input.shape for input in inputs]))
            
            input = torch.stack(inputs, dim=dim)
            return self.network(input).squeeze()


def make_forward_fn_1d(
    model: nn.Module,
    derivative_order: int = 1,
) -> list[Callable]:
    """Make a functional forward pass and gradient functions given an input model in 1-dimension

    This function creates a set of functional calls of the input model

    It returns a list of composable v-mapped version of the forward pass
    and of higher-order derivatives with respect to the inputs as
    specified by the input argument `derivative_order`

    Args:
        model (nn.Module): the model to make the functional calls for. It can be any subclass of
            a nn.Module
        derivative_order (int, optional): Up to which order return functions for computing the
            derivative of the model with respect to the inputs

    Returns:
        list[Callable]: A list of functions where each element corresponds to
            a v-mapped version of the model forward pass and its derivatives. The
            0-th element is always the forward pass and, depending on the value of
            the `derivative_order` argument, the following elements corresponds to
            the i-th order derivative function with respect to the model inputs. The
            vmap ensures efficient support for batched inputs
    """
    # notice that `functional_call` supports batched input by default
    # thus there is not need to call vmap on it, as it's instead the case
    # for the derivative calls
    def f(x: torch.Tensor, params: dict[str, torch.nn.Parameter] | tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
        
        # the functional optimizer works with parameters represented as a tuple instead
        # of the dictionary form required by the `functional_call` API 
        # here we perform the conversion from tuple to dictionary
        if isinstance(params, tuple):
            params_dict = tuple_to_dict_parameters(model, params)
        else:
            params_dict = params

        return functional_call(model, params_dict, (x, ))

    fns = []
    fns.append(f)

    dfunc = f
    for _ in range(derivative_order):

        # first compute the derivative function
        dfunc = grad(dfunc)

        # then use vmap to support batching
        dfunc_vmap = vmap(dfunc, in_dims=(0, None))

        fns.append(dfunc_vmap)

    return fns


def make_forward_fn_nd(
    model: nn.Module,
    on_variable: int,
    derivative_order: int = 1,
) -> list[Callable]:
    """Make a functional forward pass and gradient functions given an input model in n-dimensions.

    The parameters are exactly as the function above. The call to `grad` has been replaced with
    a call to the reversed mode AD `jacrev` which computes the Jacobian of the function. Notice that
    `jacrev` automatically supports batched inputs.
    """

    # notice that `functional_call` supports batched input by default
    def f(*inputs: torch.Tensor, params: dict[str, torch.nn.Parameter] | tuple[torch.nn.Parameter, ...] = None) -> torch.Tensor:
        if isinstance(params, tuple):
            params_dict = tuple_to_dict_parameters(model, params)
        else:
            params_dict = params

        return functional_call(model, params_dict, inputs)

    fns = []
    fns.append(f)

    dfunc = f
    for _ in range(derivative_order):
        
        dfunc = grad(dfunc, argnums=on_variable)
        dfunc_vmap = vmap(dfunc)
        
        fns.append(dfunc_vmap)

    return fns


def tuple_to_dict_parameters(
        model: nn.Module, params: tuple[torch.nn.Parameter, ...]
) -> OrderedDict[str, torch.nn.Parameter]:
    """Convert a set of parameters stored as a tuple into a dictionary form

    This conversion is required to be able to call the `functional_call` API which requires
    parameters in a dictionary form from the results of a functional optimization step which 
    returns the parameters as a tuple

    Args:
        model (nn.Module): the model to make the functional calls for. It can be any subclass of
            a nn.Module
        params (tuple[Parameter, ...]): the model parameters stored as a tuple
    
    Returns:
        An OrderedDict instance with the parameters stored as an ordered dictionary
    """
    keys = list(dict(model.named_parameters()).keys())
    values = list(params)
    return OrderedDict(({k:v for k,v in zip(keys, values)}))
