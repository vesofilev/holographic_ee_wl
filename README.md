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
- `ode_benchmark.py` — ODE benchmark for AdS-Schwarzschild (`h=1`, `f=1-z^4`); generates `data/ode_benchmark.npz`
- `ode_benchmark_gr.py` — ODE benchmark for AdS-Schwarzschild (AdS5)
- `ode_benchmark_gr_d3.py` — ODE benchmark for AdS4 Gubser-Rocha
- `gubser_rocha.py` — Gubser-Rocha metric functions (AdS5)
- `gubser_rocha_d3.py` — Gubser-Rocha metric functions (AdS4)

### Inverse Problem
- `ann_inverse.py` — Inverse problem: recover f(z) from S_EE(l) (AdS-Schwarzschild)
- `ann_inverse_gr_d3.py` — Inverse problem for Gubser-Rocha (S_EE only; unstable due to exact metric degeneracy — see paper)
- `ann_inverse_gr_d3_wl.py` — Three-network inverse with Wilson loop data
- `ann_inverse_noise.py` — Noise robustness tests

### Robustness of the reconstruction (Sec. 5.3)
Reproduces the multi-seed and observational-noise studies. All runs minimize
**both** observables (entanglement entropy *and* Wilson loop) — the S_EE-only
loss has an exact flat direction and must not be used alone.
- `seed_study_wl.py` — three-network (S_EE + Wilson loop) inverse across random seeds; faithful checkpointing (model + optimizer + RNG) with resume
- `seed_study_wl_extend.py` — driver to extend the seed ensemble to 500k epochs
- `seed_study_wl_noise.py` — noise injected on the **observable data only** (S_EE and V(L)), σ = 0, 1%, 5%
- `seed_study_noise_fresh.py` — noise runs launched from a clean initialization
- `analyze_a_ci.py` — 95% confidence intervals for the boundary derivative `a` vs. noise level
- `aggregate_seed_study.py` — aggregate ensemble statistics across all runs
- `plot_paper_robustness.py` — figures `a_seed_convergence.pdf`, `noise_robustness_wl.pdf`

### AdS-Schwarzschild validation inverse (Sec. 5.1)
- `seed_study_h1v2.py` — `h=1` inverse with a well-conditioned structured encoding, boundary slope fixed by physics (`a=0`); multi-seed
- `plot_h1v2_paper.py` — figures `inverse_metric.pdf`, `accuracy_summary.pdf`

### Analytical Reconstruction (Bilson-Hashimoto)
- `generate_data_hp.py` — High-precision data generation with theta-substitution
- `full_reconstruction.py` — Complete non-circular Bilson-Hashimoto reconstruction
- `reconstruct_chi.py` — Chi(r) reconstruction with AdS subtraction
- `bilson_reconstruction_v3.py` — Non-circular Bilson inversion

### Appendix verification (symbolic / numerical)
- `verify_appendix_a.py` — Appendix A: flat-direction structure `g_n = f_n - n h_n + P_n` verified order-by-order through O(r^6), plus the Step-4 numerics
- `verify_appendix_ops.py` — Appendix B: Einstein-Maxwell-dilaton field equations for the Gubser-Rocha background, dual operator dimension from the linearized scalar, and the free-vs-fixed near-boundary coefficient count
- `check_bc_universality.py` — the encoded endpoint exponents (RT `1/d`, string `1/3`) and boundary values coincide with pure AdS for every geometry in the search class, i.e. they carry no interior information

### Plotting
- `plot_a_drift.py` — Boundary derivative drift without Wilson loops
- `plot_comparison.py` — Comparison plots

## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy, Matplotlib
- SymPy (for the Appendix A/B verification scripts)

## Usage

```bash
# Forward problem: single strip width
python code/ann_forward.py

# Three-network inverse with Wilson loops
python code/ann_inverse_gr_d3_wl.py

# Multi-seed robustness study (both observables)
python code/seed_study_wl.py --seed 2

# AdS-Schwarzschild validation inverse (physics-fixed boundary slope)
python code/seed_study_h1v2.py --seed 1 --encoding structured0

# Analytical Bilson-Hashimoto reconstruction
python code/generate_data_hp.py     # generate high-precision data
python code/full_reconstruction.py  # run full reconstruction

# Appendix verification (no data required)
python code/verify_appendix_a.py
python code/verify_appendix_ops.py
python code/check_bc_universality.py
```

## License

MIT
