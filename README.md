# Source Code and Data files for parallel GMG

This repository contains the examples for the paper:

```
A Flexible, Parallel, Adaptive Geometric Multigrid Method for FEM
Thomas C. Clevenger, Timo Heister, Guido Kanschat, Martin Kronbichler
```
see https://arxiv.org/abs/1904.03317 for a preprint.

A recent version of [deal.II](https://dealii.org) is required to run the examples:
- deal.II Version 9.1 or 9.2 (large scale performance is significantly better with 9.2)
- p4est
- Trilinos

This repository contains:
- ``laplace-scaling/``: Laplace code for scaling tests
- ``elasticity-dg/``: Discontinuous Galerkin example for linear elasticity. Used in the 'three cylinder' test.
- ``laplace-amg-vs-gmg/``: Laplace with jump in viscosity: comparison against AMG. A simplified version is available as [step-50 in deal.II](https://www.dealii.org/current/doxygen/deal.II/step_50.html)
