# Fracture Tip Convergence Analysis 

## Introduction: 
This repository provides the tool for extracting stress intensity factors (SIFs) for both **mode I (opening)** and **mode II (shear)** fracture problems using various numerical discretization methods through displacement correlation methods. Models are mainly implemented based on the PorePy framework. 

The current implementation is only developed for the problems with a single horizontal and central fracture in the domain.

## Features

### Benchmark problems
Two classical benchmark problems for the frature are included: 

- **Sneddon's problem**: Mode I (opening fracture) 
- **Westergaard's problem**: Mode II (shear fracture) 

### Numerical discretization options

The following discretization methods are supported: 

- **MPSA**: multi-point stress approximation
- **TPSA**: two-point stress approximation
- **QFEM**: standard quadratic finite element method
- **QFEM-QPE**: QFEM with quarter-point elements

### Model-based workflow

- SIF extraction directly based on the solved model

### Cache-based workflow

- SIF extraction by loading proviously computed model results
- Useful for simulations when solving models is computationally expensive

### Single-scale analysis
- Estimate SIFs near fracture tips using two sets of sampling locations within the same model

### Two-scale analysis
- Estimate SIFs through sampling locations across a coarse and a fine mesh of the same problem

