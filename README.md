# Basic implementation of physics-informed neural networks (PINNs)

> :warning: This repository implements PINNs for educational purposes only and it is not
> ready for production uses. For this, please refer to other open-sources PINN libraries such as 
> [DeepXDE](https://github.com/lululxvi/deepxde) or [PINA](https://github.com/mathLab/PINA). 

Basic implementation of physics-informed neural networks (PINNs) for solving ordinary and partial differential equations. There
are few self-contained scripts in this repository:

* `basic_pinn/logistic_equation_1d.py`: Solve the logistic equation, a 1D first order ordinary differential equation.
* `basic_pinn/wave_equation_1d.py`: Solve the wave equation for a vibrating string attached on both ends. This is a 2-dimensional
    problem with one spatial dimension and one time dimension.

The utility module `basic_pinn/pinn.py` includes some common routines needed to setup a PINN.

These scripts are self-contained and they will train the PINN and plot the results. Check the code for more details
on the differential equation solved. To execute any of these scripts, we recommend to use 
the [`uv`](https://astral.sh/blog/uv) package manager:

```shell
# execute the 1D logistic equation
uv run python basic_pinn/logistic_equation_1d.py

# execute the 1D wave equation
uv run python basic_pinn/wave_equation_1d.py --batch-size 50 --learning-rate 0.0075 --num-epochs 1500
```
