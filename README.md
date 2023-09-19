# Bayesian-Equation-Discovery
A Bayesian algorithm (BED) for data driven Equation Discovery. This algorithm utilises Gaussian Process (GP) Regression and Markov Chain Monte Carlo (MCMC) with a horseshoe prior for discovering the governing equations of systems of ODE's and PDE's.


## Example
An example of using the algorithm for candidate functions of polynomials upto the 4th order and derivatives up to the 4th order and includeing interactions:
```
# Data X Response U
model = BED(X, U, p_ord=4, d_ord=4, interact=True)
```

An example output for the Van der pol oscillator with parameter value 2.5: 
```
Discovered Equation(s):
dU[0]/dt = 1.03 U[1]^1
dU[1]/dt = -1.02 U[0]^1 + 2.52 U[1]^1 + -2.50 U[0]^2*U[1]^1
```
