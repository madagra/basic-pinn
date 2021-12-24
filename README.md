# Basic implementation of physics-informed neural networks (PINNs)

Basic implementation of physics-informed neural networks for solving several ordinary and partial differential equations. There
are 3 self-contained scripts in this repository:

* `logistic_equation_1d.py`: Solve the logistic equation in 1 dimension.
* `burgers_equation_1d.py`: Solve a simple 1-dimension Burgers equation with sinusoidal initial condition.
* `wave_equation_1d.py`: Solve the 1-dimension wave equation of a string clamped of both ends.

These scripts are self-contained and they will train the PINN and plot the results. Check the code for more details
on the differential equation solved. To execute any of these scripts, within your favorite virtual environment 
manager run the following:

```
python -m pip install -r requirements.txt
python logistic_equation_1d.py
```
