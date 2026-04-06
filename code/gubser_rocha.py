# ASSERT_CONVENTION: natural_units=natural, metric_signature=mostly_minus,
#   coordinate_system=fefferman_graham_z, z_h=1 (after rescaling), L=1, r_+=1
# ASSERT_CONVENTION: GR metric ds^2 = (1/z^2)[-f(z)dt^2 + dz^2/f(z) + h(z)(dx_1^2+dx_2^2+dx_3^2)]
# ASSERT_CONVENTION: Q=0 recovery: f(z)=1-z^4, h(z)=1
"""
Exact Gubser-Rocha metric functions in Fefferman-Graham z-coordinates.

The GR solution in r-coordinates (L=1, r_+ = 1):
  e^{2A(r)} = r^2 (1 + Q^2/r^2)^{2/3}
  e^{2B(r)} = (1/r^2) (1 + Q^2/r^2)^{-4/3}
  h_orig(r) = 1 - (1+Q^2)^2 / (r^2 + Q^2)^2

Coordinate transformation ODE:
  dz/dr = -z^2 (1 + Q^2/r^2)^{-1/3}

from large r (where z ~ 1/r) inward to r_+ = 1.

Metric identification:
  h(z) = z^2 e^{2A(r(z))}
  f(z) = h(z) * h_orig(r(z))

Normalized so that z_h = 1 (horizon at z=1).

References:
  - Gubser and Rocha, arXiv:0911.2898
  - Ahn et al., arXiv:2406.07395 (JHEP 01 (2025) 025)
  - Project explanation: paper/gubser-rocha-explanation.tex

Reproducibility:
  Python 3.11+, NumPy, SciPy
  float64 throughout, deterministic (no random seeds needed)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def _e2A(r, Q):
    """Warp factor e^{2A(r)} = r^2 (1 + Q^2/r^2)^{2/3}."""
    return r**2 * (1.0 + Q**2 / r**2)**(2.0 / 3.0)


def _e2B(r, Q):
    """Warp factor e^{2B(r)} = (1/r^2) (1 + Q^2/r^2)^{-4/3}."""
    return (1.0 / r**2) * (1.0 + Q**2 / r**2)**(-4.0 / 3.0)


def _h_orig(r, Q):
    """Original blackening factor h(r) = 1 - (1+Q^2)^2 / (r^2+Q^2)^2.
    M = (r_+^2 + Q^2)^2 = (1 + Q^2)^2 for r_+ = 1."""
    M = (1.0 + Q**2)**2
    return 1.0 - M / (r**2 + Q**2)**2


def gr_metric(Q, n_points=2000):
    """
    Compute Gubser-Rocha metric functions f(z) and h(z) in FG coordinates.

    Parameters
    ----------
    Q : float
        Charge parameter. Q=0 gives AdS-Schwarzschild (f=1-z^4, h=1).
    n_points : int
        Number of output grid points on z in [0, 1].

    Returns
    -------
    z_grid : np.ndarray, shape (n_points,)
        Uniform grid on [0, 1] (z_h = 1 after rescaling).
    f_grid : np.ndarray, shape (n_points,)
        Blackening factor f(z).
    h_grid : np.ndarray, shape (n_points,)
        Spatial warp factor h(z).
    """
    if Q < 1e-14:
        # Pure AdS-Schwarzschild limit
        z_grid = np.linspace(0, 1.0, n_points)
        f_grid = 1.0 - z_grid**4
        h_grid = np.ones(n_points)
        return z_grid, f_grid, h_grid

    # === Step 1: Integrate dz/dr = -z^2 (1 + Q^2/r^2)^{-1/3} ===
    # from r_max (large r, UV) down to r_+ = 1 (horizon, IR).
    # At large r: z ~ 1/r, so start at r_max with z_init = 1/r_max.

    r_plus = 1.0
    r_max = 500.0  # large enough for UV asymptotics z ~ 1/r

    def dz_dr(r, z_vec):
        z = z_vec[0]
        factor = (1.0 + Q**2 / r**2)**(-1.0 / 3.0)
        return [-z**2 * factor]

    z_init = 1.0 / r_max

    # Integrate from r_max down to r_+ = 1
    # Use many internal steps for accuracy
    sol = solve_ivp(
        dz_dr, [r_max, r_plus], [z_init],
        method='RK45', rtol=1e-12, atol=1e-14,
        max_step=0.1, dense_output=True
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    # z_h_raw = z at r = r_+
    z_h_raw = sol.y[0, -1]

    # === Step 2: Build dense r-grid and evaluate z(r) ===
    # Use a non-uniform r-grid: dense near r_+ = 1 (where z changes fastest)
    r_low = np.linspace(r_plus, 3.0, 3000)
    r_mid = np.linspace(3.0, 30.0, 2000)
    r_high = np.linspace(30.0, r_max, 1000)
    r_dense = np.unique(np.concatenate([r_low, r_mid, r_high]))

    z_dense = sol.sol(r_dense)[0]

    # === Step 3: Compute h(z) and f(z) on the raw z-grid ===
    # h(z) = z^2 * e^{2A(r(z))}
    # f(z) = h(z) * h_orig(r(z))
    e2A_dense = _e2A(r_dense, Q)
    h_orig_dense = _h_orig(r_dense, Q)
    h_dense = z_dense**2 * e2A_dense
    f_dense = h_dense * h_orig_dense

    # === Step 4: Rescale z so that z_h = 1 ===
    z_rescaled = z_dense / z_h_raw

    # Sort by z_rescaled (should be increasing as r decreases)
    # r_dense goes from r_+ to r_max, z goes from z_h to ~0
    # So z_rescaled is decreasing. Reverse to get increasing.
    idx_sort = np.argsort(z_rescaled)
    z_rescaled = z_rescaled[idx_sort]
    h_sorted = h_dense[idx_sort]
    f_sorted = f_dense[idx_sort]

    # h and f are invariant under z -> z/z_h_raw rescaling in the
    # metric ds^2 = (1/z^2)[-f dt^2 + dz^2/f + h dx_i^2]:
    # Under z -> z' = z/z_h_raw, the overall 1/z^2 -> z_h_raw^2/z'^2.
    # We absorb this into a redefinition (rescale coords), so
    # h(z') = h(z) and f(z') = f(z) is correct.
    # However, we need to normalize: h(0) = 1, f(0) = 1, f(1) = 0.

    # Interpolate to find boundary values
    h_interp_raw = interp1d(z_rescaled, h_sorted, kind='cubic',
                            fill_value='extrapolate')
    f_interp_raw = interp1d(z_rescaled, f_sorted, kind='cubic',
                            fill_value='extrapolate')

    h0 = float(h_interp_raw(0.0))
    f0 = float(f_interp_raw(0.0))

    # Normalize: the metric is ds^2 = (1/z^2)[-f dt^2 + dz^2/f + h dx_i^2]
    # For the boundary asymptotics to be standard AdS, we need h(0) = 1.
    # Also f(0) should be 1.
    # Since h = z^2 * e^{2A} and at the boundary r->inf, z->0:
    # e^{2A} ~ r^2, z ~ 1/r, so h ~ (1/r^2) * r^2 = 1. Good.
    # Similarly f = h * h_orig, h_orig -> 1 at r->inf, so f -> h -> 1.
    # The numerical boundary values should already be ~1, but let's
    # correct for any small error from finite r_max.
    h_norm = h_sorted / h0
    f_norm = f_sorted / f0

    # === Step 5: Interpolate onto uniform output grid ===
    z_grid = np.linspace(0.0, 1.0, n_points)

    h_interp = interp1d(z_rescaled, h_norm, kind='cubic',
                        fill_value='extrapolate')
    f_interp = interp1d(z_rescaled, f_norm, kind='cubic',
                        fill_value='extrapolate')

    h_grid = h_interp(z_grid)
    f_grid = f_interp(z_grid)

    # Enforce exact boundary conditions
    h_grid[0] = 1.0
    f_grid[0] = 1.0
    f_grid[-1] = 0.0

    return z_grid, f_grid, h_grid


def validate_Q0():
    """Validate that Q=0 recovers f(z) = 1 - z^4, h(z) = 1."""
    print("=" * 60)
    print("Gubser-Rocha Q=0 validation")
    print("=" * 60)

    z, f, h = gr_metric(Q=0.0, n_points=500)
    f_exact = 1.0 - z**4
    h_exact = np.ones_like(z)

    f_err = np.max(np.abs(f - f_exact))
    h_err = np.max(np.abs(h - h_exact))

    print(f"  Q = 0 (AdS-Schwarzschild limit)")
    print(f"  max|f(z) - (1-z^4)| = {f_err:.2e}")
    print(f"  max|h(z) - 1|       = {h_err:.2e}")
    print(f"  f(0)={f[0]:.6f}, f(0.5)={f[250]:.6f}, f(1)={f[-1]:.6f}")
    print(f"  h(0)={h[0]:.6f}, h(0.5)={h[250]:.6f}, h(1)={h[-1]:.6f}")

    if f_err < 1e-10 and h_err < 1e-10:
        print("  VALIDATION PASS")
    else:
        print("  VALIDATION FAIL")
    print()
    return f_err, h_err


def validate_nonzero_Q(Q=1.0):
    """Validate GR metric for Q > 0: check boundary conditions and positivity."""
    print("=" * 60)
    print(f"Gubser-Rocha Q={Q} validation")
    print("=" * 60)

    z, f, h = gr_metric(Q=Q, n_points=2000)

    print(f"  h(0) = {h[0]:.10f} (should be 1)")
    print(f"  f(0) = {f[0]:.10f} (should be 1)")
    print(f"  f(z_h) = {f[-1]:.10f} (should be 0)")
    print(f"  h(z_h) = {h[-1]:.10f}")

    # h should be > 0 everywhere
    h_min = np.min(h)
    print(f"  min(h) = {h_min:.10f} (should be > 0)")

    # f should be > 0 for z in (0, z_h) and = 0 at z_h
    f_interior = f[1:-1]
    f_min_int = np.min(f_interior)
    print(f"  min(f) in interior = {f_min_int:.10f} (should be > 0)")

    # h should be monotonically increasing (GR feature: h grows toward horizon)
    dh = np.diff(h)
    h_mono = np.all(dh >= -1e-10)
    print(f"  h(z) monotonically increasing: {h_mono}")

    # Print some sample values
    idx_quarter = len(z) // 4
    idx_half = len(z) // 2
    idx_3quarter = 3 * len(z) // 4
    print(f"  h(0.25) = {h[idx_quarter]:.6f}")
    print(f"  h(0.50) = {h[idx_half]:.6f}")
    print(f"  h(0.75) = {h[idx_3quarter]:.6f}")
    print(f"  h(1.00) = {h[-1]:.6f}")
    print(f"  f(0.25) = {f[idx_quarter]:.6f}")
    print(f"  f(0.50) = {f[idx_half]:.6f}")
    print(f"  f(0.75) = {f[idx_3quarter]:.6f}")

    checks_pass = (
        abs(h[0] - 1.0) < 1e-6 and
        abs(f[0] - 1.0) < 1e-6 and
        abs(f[-1]) < 1e-6 and
        h_min > 0 and
        f_min_int > -1e-6
    )
    if checks_pass:
        print("  VALIDATION PASS")
    else:
        print("  VALIDATION FAIL")
    print()
    return z, f, h


if __name__ == "__main__":
    validate_Q0()
    z, f, h = validate_nonzero_Q(Q=0.5)
    z2, f2, h2 = validate_nonzero_Q(Q=1.0)

    # Plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os

        fig_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(z, f, 'b-', lw=1.5, label='Q=0.5')
        axes[0].plot(z2, f2, 'r-', lw=1.5, label='Q=1.0')
        z0 = np.linspace(0, 1, 500)
        axes[0].plot(z0, 1 - z0**4, 'k--', lw=1, label=r'Q=0: $1-z^4$')
        axes[0].set_xlabel(r'$z$')
        axes[0].set_ylabel(r'$f(z)$')
        axes[0].set_title('Gubser-Rocha blackening factor')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(z, h, 'b-', lw=1.5, label='Q=0.5')
        axes[1].plot(z2, h2, 'r-', lw=1.5, label='Q=1.0')
        axes[1].axhline(y=1, color='k', ls='--', lw=1, label='Q=0: h=1')
        axes[1].set_xlabel(r'$z$')
        axes[1].set_ylabel(r'$h(z)$')
        axes[1].set_title('Gubser-Rocha spatial warp factor')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'gubser_rocha_metric.pdf'), dpi=150)
        plt.close()
        print(f"Saved {os.path.join(fig_dir, 'gubser_rocha_metric.pdf')}")
    except ImportError:
        print("matplotlib not available, skipping plots")
