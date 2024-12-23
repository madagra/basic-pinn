
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor


def plot_1d_solution(
    x_eval: Tensor, 
    x_sample: Tensor,
    f_eval: Tensor, 
    analytical_sol_fn: Callable, 
    loss_evolution: Tensor,
    show: bool = True,
) -> Figure:
    """
    Plot the 1D solution of the logistic equation.
    Args:
        x_eval (torch.Tensor): Evaluation points for the solution.
        f_eval (torch.Tensor): Evaluated solution at the evaluation points.
        analytical_sol_fn (callable): Analytical solution function.
        x_sample_np (numpy.ndarray): Sample training points.
        loss_evolution (list): List of loss values at each epoch.
    """
    x_eval_np = x_eval.detach().numpy()
    x_sample_np = x_sample.detach().numpy()

    fig, ax = plt.subplots()

    ax.scatter(x_sample_np, analytical_sol_fn(x_sample_np), color="red", label="Sample training points")
    ax.plot(x_eval_np, f_eval.detach().numpy(), label="PINN final solution")
    ax.plot(
        x_eval_np,
        analytical_sol_fn(x_eval_np),
        label="Analytic solution",
        color="green",
        alpha=0.75,
    )
    ax.set(title="Equation solved with NNs", xlabel="t", ylabel="f(t)")
    ax.legend()

    fig, ax = plt.subplots()
    ax.semilogy(loss_evolution)
    ax.set(title="Loss evolution", xlabel="# epochs", ylabel="Loss")
    ax.legend()

    if show:
        plt.show()

    return fig