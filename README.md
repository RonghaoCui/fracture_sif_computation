# Fracture Tip Convergence Analysis

## Introduction: 
Evaluate the near-tip singularity order (exponent) and Stress Intensity Factor (SIF) for Sneddon’s problem through sampling displacement jump along the fracture, with single-scale and two-scale options. 

Currently, this script is only developed for a single straight fracture.

## Feature: 
- **Single-scale option**: Estimate the singularity order near fracture tips using two sets of sampling locations within the same model.
- **Two-scale option**: Estimate the singularity order through sampling locations across a coarse and a fine mesh of the same problem.
- **Sampling modes**:
  - `{"mode": "fixed", "steps": k}` — sample at the *k*-th cell away from the fracture tip.
  - `{"mode": "refinement", "level": L}` — sample based on refinement level *L* (internally converted to a step count).
