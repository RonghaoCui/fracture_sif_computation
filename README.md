# Computation of Fracture Stress Intensity Factors

## Introduction
This repository provides numerical models and run scripts used to reproduce the results presented in the manuscript titled *A scaling law for numerical computation of fracture stress intensity factors*.

Stress intensity factors (SIFs) are extracted for both **mode-I (opening)** and **mode-II (sliding)** fracture problems using two-point displacement correlation in three numerical discretization methods: 

- **MPSA**: Multi-point stress approximation
- **QFEM**: Standard quadratic finite element method
- **QFEM_QPE**: QFEM with quarter-point elements 

The current computation tool only supports a single central fracture with different orientations in a 2D domain. All computational geometries are generated from [PorePy](https://github.com/pmgbergen/porepy).

## Benchmark problems
Four fracture benchmarks are included: 

- **Sneddon mode-I problem ('so')**: fracture opening induced by the constant pressure.
- **Westergaard mode-I problem ('wo')**: fracture opening induced by the uniform far-field normal stress. 
- **Westergaard mode-II problem ('ws')**: fracture sliding induced by the uniform far-field in-plane shear stress. 
- **Westergaard mixed-mode problem ('wm')**: fracture opening and sliding induced by the uniform far-field normal and in-plane shear stresses.

The scripts in [solutions](./solutions/) run all benchmark models using MPSA, QFEM, and QFEM_QPE and save the resulting plots under figures/solutions/.

## Numerical analyses

### Global convergence analysis

The scripts in [sol_convergence](./sol_convergence/) compute relative $L^2$ errors of numerical displacement and displacement jump fields against reference solutions with grid refinement.

### Cache-based workflow for SIF convergence analysis

The scripts in [cache_modes](./cache_modes/) implement the following workflow:

- Run benchmark models (using MPSA, QFEM, and QFEM_QPE) and cache the computed solutions.
- Load previously computed model solutions and evaluate relative $L^2$ errors of SIFs along the fracture.
- Identify the scaling law with the grid refinement.
- Carry out convergence analysis for SIFs using two sampling strategies, including nearest-tip sampling and scaling-based sampling.

### Manuscript results
- Run all scripts from the repository root so that the relative input and output paths can resolve correctly.
- Run the scripts in [plotfigs](./plotfigs/) to reproduce the manuscript figures.

## Referenced resources
- The MPSA simulations use the weak-symmetry formulation implemented in [PorePy](https://github.com/pmgbergen/porepy). 
- The implementations of QFEM and QFEM_QPE are adapted from [Dang et al. (2022)](https://doi.org/10.1016/j.ijrmms.2022.105248), with modifications for this work.
