
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import torch
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
    Plot the solution of a 1-dimension differential equation

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


def animate_2d_solution(
    x_eval: Tensor,
    t_eval: Tensor,
    opt_params: tuple,
    fn: Callable,
    show: bool = True
) -> tuple[Figure, FuncAnimation]:
    """
    Animate the solution of a 2-dimension problem in time and space

    Args:
        x_eval (Tensor): Evaluation points for the spatial dimension.
        t_eval (Tensor): Evaluation points for the time dimension.
        opt_params (tuple): Optimal parameters computed after the training procedure.
        fn (Callable): The function to animate.
        show (bool, optional): Whether to display the animation. Defaults to True.

    Returns:
        tuple[Figure, FuncAnimation]: A tuple containing the matplotlib Figure and FuncAnimation objects.
    """
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)

    ax.set(title="Equation solved with NNs", xlabel="x", ylabel="u(x,t)")

    def init() -> tuple:
        ax.set_xlim(x_eval[0].item(), x_eval[-1].item())
        y_values = [fn(x_eval, t * torch.ones_like(x_eval), params=opt_params).detach().numpy() for t in t_eval]
        ax.set_ylim(min(map(min, y_values)), max(map(max, y_values)))
        return line,

    def animate(frame: int) -> tuple:
        t = t_eval[frame]
        y = fn(x_eval, t * torch.ones_like(x_eval), params=opt_params)
        line.set_data(x_eval.detach().numpy(), y.detach().numpy())
        return line,

    ani = FuncAnimation(
        fig, animate, frames=len(t_eval), init_func=init, blit=True, interval=200, repeat=False
    )

    if show:
        plt.show()

    return fig, ani
