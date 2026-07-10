# Gravitational Wave Source Parameter Estimation using Simulation-Based Inference

This project implements a deep learning pipeline for estimating the physical parameters of binary black hole mergers directly from noisy gravitational-wave signals.

Using realistic waveforms generated with **PyCBC (IMRPhenomD)** and **BayesFlow's Neural Posterior Estimation (NPE)**, the model learns to infer six astrophysical parameters—component masses, aligned spins, luminosity distance, and orbital inclination—from simulated LIGO detector strain data.

Unlike traditional Bayesian sampling methods (e.g., MCMC or Nested Sampling), this approach uses **amortized simulation-based inference**, enabling rapid posterior estimation after training while maintaining probabilistic uncertainty estimates.

## Key Features

- Generated realistic binary black hole merger waveforms using **PyCBC**
- Simulated colored Advanced LIGO detector noise using the **aLIGOZeroDetHighPower PSD**
- Applied signal whitening and preprocessing for robust learning
- Built a **1D CNN Summary Network** to extract informative features from time-series data
- Trained a **Normalizing Flow (BayesFlow)** for Bayesian posterior estimation
- Evaluated model calibration using **Simulation-Based Calibration (SBC)** and posterior diagnostics
- Developed as part of the **Simulation-Based Inference** course at **TU Dortmund University**

## Technologies

- Python
- PyCBC
- BayesFlow
- JAX
- Keras 3
- NumPy
- Matplotlib
- Bayesian Deep Learning
- Simulation-Based Inference
