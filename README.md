# Manifold Optimization


Generalization on manifolds of some well known ML optimization algorithms. The implementation
is similar to and inspired by the toolbox Pymanopt, please visit the official github page
https://github.com/pymanopt/pymanopt for more informations.

In this repository I consider the particular case of a manifold which is the product of the
Euclidean manifold and the manifold of positive definite matrices. Indeed this manifold
corresponds to the parameter space of a Riemann-Theta Boltzmann Machine and therefore what I
am trying to do is developing a new training algorithm for the model. More informations on
RTBMs can be found here https://github.com/RiemannAI/theta.

## Requirements:

- theta 0.0.1

