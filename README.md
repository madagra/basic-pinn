# Basic implementation of physics-informed neural networks (PINNs)

Basic implementation of physics-informed neural networks (PINNs) for solving several ordinary and partial differential equations. There
are few self-contained scripts in this repository:

* `logistic_equation.py`: Solve the logistic equation, a 1D first order ordinary differential equation.

The utility module `pinn.py` includes some common routines needed to setup a PINN.

These scripts are self-contained and they will train the PINN and plot the results. Check the code for more details
on the differential equation solved. To execute any of these scripts, within your favorite virtual environment 
manager run the following:

```
python -m venv .env
source .env/bin/activate
python -m pip install -r requirements.txt
python logistic_equation.py
```
