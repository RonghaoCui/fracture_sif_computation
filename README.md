# Convergence Analysis of Stress Intensity Factors

## Introduction: 
This repository provides numerical models and run scripts used to reproduce the results presented in the manuscript titled *A scaling law for numerical computation of fracture stress intensity factors*.

Stress intensity factors (SIFs) are extracted for both **mode-I (opening)** and **mode-II (sliding)** fracture problems through two-point displacement correlation methods in three numerical discretization methods: 

- **MPSA**: Multi-point stress approximation
- **QFEM**: Standard quadratic finite element method
- **QFEM_QPE**: QFEM with quarter-point elements 

The current tool is only developed for the problems with a single central fracture with multiple orientations in the 2D domain. All of computational geometries are obtained from [PorePy](https://github.com/pmgbergen/porepy). 

## Benchmark problems
Four fracture benchmarks are included: 

- **Sneddon mode-I problem ('so')**: fracture opening induced by the constant pressure.
- **Westergaard mode-I problem ('wo')**: fracture opening by the uniform far-field normal stress. 
- **Westergaard mode-II problem ('ws')**: fracture sliding by the uniform far-field in-plane shear stress. 
- **Westergaard mixed-mode problem ('wm')**: fracture opening and sliding by the uniform far-field normal and in-plane shear stresses.

Solutions of all benchmark models are located in [solutions](./solutions/).

## Numerical analyses

### Self-convergence analysis

The scripts in [convergence](./convergence/) evaluate self-convergence of displacement and displacement jump solutions by relative $L^2$ errors.

### Cache-based workflow for SIF convergence analysis

The scripts in [cache_modes](./cache_modes/) implement the following workflow:

- Run benchmark models (by MPSA, QFEM, and QFEM_QPE) and cache computed solutions.
- Load proviously computed model solutions and evaluate relative $L^2$ errors of SIFs along the fracture.
- Identify the scaling law with the grid refinement.
- Carry out convergence analysis for SIFs by two sampling strategies, including nearest-tip sampling and scaling-based sampling.

## Referenced resources
- The MPSA simulations use the weak-symmetry formulation implemented in [PorePy](https://github.com/pmgbergen/porepy). 
- The implementations of QFEM and QFEM_QPE are partially based on [Dang et al. (2022)](https://doi.org/10.1016/j.ijrmms.2022.105248).
- The directory of [mdamr](./mdamr/) includes utilities for evaluating solution differences on non-matching simplex grids, developed by [Varela et al. (2025)](https://doi.org/10.48550/arXiv.2512.09087).
