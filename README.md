# Holographic Entanglement Entropy, Wilson Loops, and Neural Networks

PyTorch code accompanying the paper:

**"Holographic entanglement entropy, Wilson loops, and neural networks"**
by Veselin G. Filev

## Overview

This repository contains the code for computing Ryu-Takayanagi minimal surfaces and reconstructing bulk geometry from boundary entanglement entropy and Wilson loop data using artificial neural networks.

## Code Structure

### Forward Problem
- `ann_forward.py` — ANN with area-as-loss for single strip widths and conditional network z(x,l)
- `utils.py` — Grid generation and common utilities
- `pretrain_exact.py` — Pre-training on exact ODE solutions

### ODE Benchmarks
- `ode_benchmark_gr.py` — ODE benchmark for AdS-Schwarzschild
- `ode_benchmark_gr_d3.py` — ODE benchmark for AdS4 Gubser-Rocha
- `gubser_rocha.py` — Gubser-Rocha metric functions (AdS5)
- `gubser_rocha_d3.py` — Gubser-Rocha metric functions (AdS4)

### Inverse Problem
- `ann_inverse.py` — Inverse problem: recover f(z) from S_EE(l) (AdS-Schwarzschild)
- `ann_inverse_gr_d3.py` — Inverse problem for Gubser-Rocha (S_EE only; unstable due to exact metric degeneracy — see paper)
- `ann_inverse_gr_d3_wl.py` — Three-network inverse with Wilson loop data
- `ann_inverse_noise.py` — Noise robustness tests

### Analytical Reconstruction (Bilson-Hashimoto)
- `generate_data_hp.py` — High-precision data generation with theta-substitution
- `full_reconstruction.py` — Complete non-circular Bilson-Hashimoto reconstruction
- `reconstruct_chi.py` — Chi(r) reconstruction with AdS subtraction
- `bilson_reconstruction_v3.py` — Non-circular Bilson inversion
- `bilson_reconstruction_v2.py` — Earlier version (for reference)

### Plotting
- `plot_a_drift.py` — Boundary derivative drift without Wilson loops
- `plot_comparison.py` — Comparison plots

## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy, Matplotlib

## Usage

```bash
# Forward problem: single strip width
python code/ann_forward.py

# Three-network inverse with Wilson loops
python code/ann_inverse_gr_d3_wl.py

# Analytical Bilson-Hashimoto reconstruction
python code/generate_data_hp.py  # generate high-precision data
python code/full_reconstruction.py  # run full reconstruction
```

## License

MIT
