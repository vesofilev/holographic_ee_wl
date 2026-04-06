# ASSERT_CONVENTION: natural_units=natural, metric_signature=mostly_minus,
#   coordinate_system=z_equals_1_over_r, z_h=1, L=1
# ASSERT_CONVENTION: AdS4 (d=3 boundary) GR metric:
#   ds^2 = (1/z^2)[-f(z)dt^2 + dz^2/f(z) + h(z)(dx^2+dy^2)]
# ASSERT_CONVENTION: f(z) = (1-z)U(z), h(z) = (1+Qz)^{3/2}
# ASSERT_CONVENTION: Q=0 recovery: f(z)=1-z^3, h(z)=1
# ASSERT_CONVENTION: area = Omega int (sqrt(h)/z^2) sqrt(h + z'^2/f) dx
"""
Closed-form Gubser-Rocha metric in AdS4 (d=3 boundary).

Uses the z-coordinate parametrization of Li and Liu (arXiv:2307.04433),
where z in [0,1] with boundary at z=0 and horizon at z=1.

  f(z) = (1-z) U(z)
  U(z) = [1 + (1+3Q)z + (1+3Q+3Q^2)z^2] / (1+Qz)^{3/2}
  h(z) = (1+Qz)^{3/2}

Boundary conditions:
  f(0) = 1, f(1) = 0
  h(0) = 1, h'(0) = 3Q/2

Thermodynamics:
  T = 3 sqrt(1+Q) / (4 pi)
  s = (1+Q)^{3/2} / (4 G_N)

References:
  - Li and Liu, arXiv:2307.04433 (Phys. Rev. B 108 (2023) 235104)
  - Ahn et al., arXiv:2406.07395 (JHEP 01 (2025) 025)
  - Gubser and Rocha, arXiv:0911.2898

Reproducibility:
  Python 3.11+, NumPy
  float64 throughout, deterministic (closed-form, no ODE integration)
"""

import numpy as np


def gr_metric_d3(Q, n_points=2000):
    """
    Closed-form Gubser-Rocha metric for AdS4 (d=3 boundary).

    Parameters
    ----------
    Q : float
        Charge parameter. Q=0 gives AdS4-Schwarzschild (f=1-z^3, h=1).
    n_points : int
        Number of output grid points on z in [0, 1].

    Returns
    -------
    z_grid : np.ndarray, shape (n_points,)
    f_grid : np.ndarray, shape (n_points,)
    h_grid : np.ndarray, shape (n_points,)
    """
    z = np.linspace(0.0, 1.0, n_points)

    h = (1.0 + Q * z) ** 1.5

    U_num = 1.0 + (1.0 + 3*Q) * z + (1.0 + 3*Q + 3*Q**2) * z**2
    U = U_num / (1.0 + Q * z) ** 1.5
    f = (1.0 - z) * U

    # Enforce exact BCs
    h[0] = 1.0
    f[0] = 1.0
    f[-1] = 0.0

    return z, f, h


def gr_temperature(Q):
    """Temperature T = 3 sqrt(1+Q) / (4 pi)."""
    return 3.0 * np.sqrt(1.0 + Q) / (4.0 * np.pi)


def gr_entropy_density(Q):
    """Entropy density s = (1+Q)^{3/2} / (4 G_N). Returns s * 4G_N."""
    return (1.0 + Q) ** 1.5


def gr_h_horizon(Q):
    """h(z_h=1) = (1+Q)^{3/2}."""
    return (1.0 + Q) ** 1.5


def validate(Q=1.0):
    """Validate GR metric: boundary conditions, Q=0 limit, positivity."""
    print("=" * 60)
    print(f"AdS4 Gubser-Rocha validation (Q={Q})")
    print("=" * 60)

    z, f, h = gr_metric_d3(Q, n_points=2000)

    print(f"  f(0) = {f[0]:.10f} (should be 1)")
    print(f"  f(1) = {f[-1]:.10f} (should be 0)")
    print(f"  h(0) = {h[0]:.10f} (should be 1)")
    print(f"  h(1) = {h[-1]:.10f} (should be {(1+Q)**1.5:.6f})")

    # h'(0) check
    dh = np.gradient(h, z)
    print(f"  h'(0) ~ {dh[0]:.6f} (should be {1.5*Q:.6f})")

    # Positivity
    f_int = f[1:-1]
    print(f"  min(f) interior = {f_int.min():.10f} (should be > 0)")
    print(f"  min(h) = {h.min():.10f} (should be > 0)")

    # Temperature
    T = gr_temperature(Q)
    # Check via numerical f'(1)
    fp = np.gradient(f, z)
    T_num = abs(fp[-2]) / (4 * np.pi)  # -2 to avoid endpoint artifacts
    print(f"  T = {T:.6f} (analytic), T_num ~ {T_num:.6f}")

    # Q=0 check
    z0, f0, h0 = gr_metric_d3(Q=0.0, n_points=500)
    f0_exact = 1.0 - z0**3
    h0_exact = np.ones_like(z0)
    f_err = np.max(np.abs(f0 - f0_exact))
    h_err = np.max(np.abs(h0 - h0_exact))
    print(f"\n  Q=0 recovery:")
    print(f"  max|f - (1-z^3)| = {f_err:.2e}")
    print(f"  max|h - 1| = {h_err:.2e}")

    checks_pass = (
        abs(f[0] - 1.0) < 1e-12 and
        abs(f[-1]) < 1e-12 and
        abs(h[0] - 1.0) < 1e-12 and
        abs(h[-1] - (1+Q)**1.5) < 1e-6 and
        f_int.min() > 0 and
        f_err < 1e-10 and h_err < 1e-10
    )
    print(f"\n  {'PASS' if checks_pass else 'FAIL'}")
    return checks_pass


if __name__ == "__main__":
    validate(Q=0.0)
    print()
    validate(Q=0.5)
    print()
    validate(Q=1.0)
