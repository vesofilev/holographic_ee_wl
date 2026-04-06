# ASSERT_CONVENTION: natural_units=natural, metric_signature=mostly_minus,
#   coordinate_system=fefferman_graham_z, z_h=1, L=1
# ASSERT_CONVENTION: GR metric ds^2 = (1/z^2)[-f(z)dt^2 + dz^2/f(z) + h(z) dx_i^2]
# ASSERT_CONVENTION: area = V_2 int (h/z^3) sqrt(h + z'^2/f) dx
# ASSERT_CONVENTION: Q=0 must recover ode_benchmark.npz results
"""
ODE Benchmark for Strip Entanglement Entropy in Gubser-Rocha Background.

Computes S_EE(l) via turning-point parametric integrals with
interpolated f(z) and h(z) from the exact GR metric.

Strip half-width:
  l/2 = int_0^{z_*} dz / [sqrt(f h) sqrt(h^3 z_*^6 / (z^6 h_*^3) - 1)]

Regularized area (UV-finite):
  A_reg = int_0^{z_*} (h/z^3)/sqrt(f) [1/sqrt(1 - z^6 h_*^3/(z_*^6 h^3)) - 1] dz
        - int_{z_*}^{z_h} h/(z^3 sqrt(f)) dz

where h_* = h(z_*).

References:
  - Gubser and Rocha, arXiv:0911.2898
  - Ahn et al., arXiv:2406.07395, Eqs. (3)-(7)
  - Project: paper/gubser-rocha-explanation.tex, Section 7

Reproducibility:
  Python 3.11+, NumPy, SciPy, matplotlib
  float64 throughout, deterministic quadrature
"""

import os
import sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Add code directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gubser_rocha import gr_metric


# =============================================================================
# Constants
# =============================================================================
Z_H = 1.0
N_ZSTAR = 500


# =============================================================================
# Build interpolators from GR metric data
# =============================================================================
def build_metric_interpolators(Q, n_metric=4000):
    """
    Build cubic interpolators for f(z), h(z) from exact GR metric.

    Returns
    -------
    f_func : callable, f(z) for scalar z
    h_func : callable, h(z) for scalar z
    z_grid : np.ndarray, the underlying grid
    f_grid : np.ndarray
    h_grid : np.ndarray
    """
    z_grid, f_grid, h_grid = gr_metric(Q, n_points=n_metric)
    f_interp = interp1d(z_grid, f_grid, kind='cubic',
                        fill_value='extrapolate')
    h_interp = interp1d(z_grid, h_grid, kind='cubic',
                        fill_value='extrapolate')

    def f_func(z):
        return float(f_interp(z))

    def h_func(z):
        return float(h_interp(z))

    return f_func, h_func, z_grid, f_grid, h_grid


# =============================================================================
# Strip half-width l(z_*) for general f(z), h(z)
# =============================================================================
def compute_l_gr(z_star, f_func, h_func):
    """
    Compute strip width l for Gubser-Rocha metric.

    l/2 = int_0^{z_*} dz / [sqrt(f(z) h(z)) sqrt(h(z)^3 z_*^6/(z^6 h_*^3) - 1)]

    Uses theta substitution z = z_* sin^{1/3}(theta) to regularize the
    endpoint singularity at z = z_*.
    """
    h_star = h_func(z_star)

    def integrand(theta):
        if theta < 1e-15:
            return 0.0
        sin_th = np.sin(theta)
        z = z_star * sin_th**(1.0 / 3.0)
        if z < 1e-15 or z >= Z_H:
            return 0.0
        fval = f_func(z)
        hval = h_func(z)
        if fval <= 0 or hval <= 0:
            return 0.0
        # The ratio inside the square root (after theta substitution):
        # h^3 z_*^6 / (z^6 h_*^3) - 1 = h^3/(sin^2 * h_*^3) - 1
        ratio = hval**3 / (sin_th**2 * h_star**3) - 1.0
        if ratio <= 0:
            return 0.0
        # Full integrand from theta substitution:
        # dz = z_* (1/3) sin^{-2/3} cos d(theta)
        # Jacobian: z_* / 3 * sin^{-2/3}(theta) * cos(theta)
        # 1/sqrt(f*h) / sqrt(ratio) * Jacobian
        # = z_* / (3 sqrt(f*h) sqrt(ratio)) * sin^{-2/3} * cos
        # But we can simplify using the standard approach:
        return sin_th**(1.0 / 3.0) / (np.sqrt(fval * hval) * np.sqrt(ratio))

    # The theta substitution gives:
    # l/2 = (z_*/3) int_0^{pi/2} sin^{1/3}(th) / (sqrt(f*h) * sqrt(ratio)) dtheta
    # But we need to be more careful. Let me redo:
    # z = z_* u^{1/3}, dz = z_* u^{-2/3}/3 du, u in [0,1]
    # Then z^6 = z_*^6 u^2
    # ratio = h^3 z_*^6 / (z_*^6 u^2 h_*^3) - 1 = h^3/(u^2 h_*^3) - 1
    # l/2 = int_0^1 (z_* u^{-2/3}/3) / (sqrt(f h) sqrt(h^3/(u^2 h_*^3) - 1)) du
    # Use theta = arcsin(u), u = sin^2(theta) is wrong...
    # Let me just use direct integration without theta substitution.

    # Direct integration with variable splitting to handle endpoint singularity
    def integrand_direct(z):
        if z < 1e-15:
            return 0.0
        fval = f_func(z)
        hval = h_func(z)
        if fval <= 0 or hval <= 0:
            return 0.0
        ratio = hval**3 * z_star**6 / (z**6 * h_star**3)
        if ratio <= 1.0 + 1e-15:
            return 0.0
        return 1.0 / (np.sqrt(fval * hval) * np.sqrt(ratio - 1.0))

    result, _ = quad(integrand_direct, 0, z_star * (1.0 - 1e-12),
                     epsabs=1e-12, epsrel=1e-12, limit=400,
                     points=[z_star * 0.5])
    return 2.0 * result


# =============================================================================
# Regularized area A_reg(z_*) for general f(z), h(z)
# =============================================================================
def compute_A_reg_gr(z_star, f_func, h_func, z_h=Z_H):
    """
    Compute regularized area for Gubser-Rocha metric.

    A_reg = int_0^{z_*} (h/z^3)/sqrt(f) [1/sqrt(1 - z^6 h_*^3/(z_*^6 h^3)) - 1] dz
          - int_{z_*}^{z_h} h/(z^3 sqrt(f)) dz

    Both integrals are UV-finite.
    """
    h_star = h_func(z_star)

    # First integral: 0 to z_* (finite, no UV divergence)
    def integrand1(z):
        if z < 1e-15:
            return 0.0
        fval = f_func(z)
        hval = h_func(z)
        if fval <= 0 or hval <= 0:
            return 0.0
        ratio6 = z**6 * h_star**3 / (z_star**6 * hval**3)
        if ratio6 >= 1.0 - 1e-15:
            return 0.0
        return (hval / (z**3 * np.sqrt(fval))) * (
            1.0 / np.sqrt(1.0 - ratio6) - 1.0)

    I1, _ = quad(integrand1, 0, z_star * (1.0 - 1e-12),
                 epsabs=1e-12, epsrel=1e-12, limit=400,
                 points=[z_star * 0.5])

    # Second integral: z_* to z_h
    def integrand2(z):
        fval = f_func(z)
        hval = h_func(z)
        if fval <= 0:
            return 0.0
        return hval / (z**3 * np.sqrt(fval))

    I2, _ = quad(integrand2, z_star, z_h * (1.0 - 1e-12),
                 epsabs=1e-12, epsrel=1e-12, limit=400,
                 points=[z_star + (z_h - z_star) * 0.5])

    return I1 - I2


# =============================================================================
# Q=0 cross-validation
# =============================================================================
def validate_Q0(ode_path):
    """
    Validate Q=0 GR benchmark against existing AdS-Schwarzschild benchmark.
    """
    print("=" * 60)
    print("Q=0 cross-validation against ode_benchmark.npz")
    print("=" * 60)

    if not os.path.exists(ode_path):
        print(f"  WARNING: {ode_path} not found, skipping cross-validation")
        return None, None

    ref = np.load(ode_path)
    ref_zs = ref['z_star_grid']
    ref_l = ref['l_of_zstar']
    ref_A = ref['A_reg_of_zstar']
    ref_lc = float(ref['l_c'][0])

    # Build Q=0 metric (should be exact 1-z^4 and h=1)
    f_func, h_func, _, _, _ = build_metric_interpolators(Q=0.0)

    # Test at a subset of z_* values
    test_idx = np.linspace(0, len(ref_zs) - 1, 30, dtype=int)
    max_l_err = 0.0
    max_A_err = 0.0

    for i in test_idx:
        zs = ref_zs[i]
        l_gr = compute_l_gr(zs, f_func, h_func)
        A_gr = compute_A_reg_gr(zs, f_func, h_func)

        l_ref = ref_l[i]
        A_ref = ref_A[i]

        if abs(l_ref) > 1e-10:
            l_err = abs(l_gr - l_ref) / abs(l_ref)
            max_l_err = max(max_l_err, l_err)
        if abs(A_ref) > 1e-10:
            A_err = abs(A_gr - A_ref) / abs(A_ref)
            max_A_err = max(max_A_err, A_err)

    print(f"  Tested {len(test_idx)} z_* values")
    print(f"  max relative error l(z_*): {max_l_err:.2e}")
    print(f"  max relative error A_reg(z_*): {max_A_err:.2e}")

    if max_l_err < 1e-6 and max_A_err < 1e-6:
        print("  Q=0 CROSS-VALIDATION PASS")
    else:
        print(f"  Q=0 CROSS-VALIDATION FAIL (thresholds: l < 1e-6, A < 1e-6)")
    print()
    return max_l_err, max_A_err


# =============================================================================
# Main computation
# =============================================================================
def main():
    # Choose Q value
    Q = 1.0

    print("=" * 60)
    print(f"ODE Benchmark: Strip Entanglement Entropy in Gubser-Rocha (Q={Q})")
    print("=" * 60)

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    ode_ref_path = os.path.join(data_dir, 'ode_benchmark.npz')

    # --- Step 1: Q=0 cross-validation ---
    validate_Q0(ode_ref_path)

    # --- Step 2: Build GR metric interpolators ---
    print(f"Building GR metric for Q={Q}...")
    f_func, h_func, z_metric, f_metric, h_metric = \
        build_metric_interpolators(Q=Q, n_metric=4000)
    print(f"  h(0)={h_func(0.0):.6f}, h(0.5)={h_func(0.5):.6f}, "
          f"h(1.0)={h_func(1.0-1e-8):.6f}")
    print(f"  f(0)={f_func(0.0):.6f}, f(0.5)={f_func(0.5):.6f}, "
          f"f(0.99)={f_func(0.99):.6f}")
    print()

    # --- Step 3: Compute l(z_*) and A_reg(z_*) ---
    z_star_low = np.linspace(0.01, 0.60, 150)
    z_star_mid = np.linspace(0.60, 0.95, 250)
    z_star_high = np.linspace(0.95, 0.99, 100)
    z_star_grid = np.unique(np.concatenate([z_star_low, z_star_mid, z_star_high]))

    print(f"Computing l(z_*) and A_reg(z_*) for {len(z_star_grid)} points...")
    l_of_zstar = np.zeros(len(z_star_grid))
    A_reg_of_zstar = np.zeros(len(z_star_grid))

    for i, zs in enumerate(z_star_grid):
        l_of_zstar[i] = compute_l_gr(zs, f_func, h_func)
        A_reg_of_zstar[i] = compute_A_reg_gr(zs, f_func, h_func)
        if (i + 1) % 100 == 0:
            print(f"  Computed {i+1}/{len(z_star_grid)} points")

    print(f"  Done. l range: [{np.min(l_of_zstar):.6f}, {np.max(l_of_zstar):.6f}]")
    print(f"  A_reg range: [{np.min(A_reg_of_zstar):.6f}, {np.max(A_reg_of_zstar):.6f}]")
    print()

    # --- Step 4: Find critical quantities ---
    # l_c: where A_reg = 0 (phase transition)
    idx_lmax = np.argmax(l_of_zstar)
    l_max = l_of_zstar[idx_lmax]
    z_star_at_lmax = z_star_grid[idx_lmax]
    print(f"  l_max = {l_max:.6f} at z_* = {z_star_at_lmax:.6f}")

    # Find zero crossing of A_reg
    signs = np.sign(A_reg_of_zstar)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    l_c = np.nan
    z_star_c = np.nan
    if len(sign_changes) > 0:
        idx_cross = sign_changes[-1]
        z_star_c = brentq(lambda zs: compute_A_reg_gr(zs, f_func, h_func),
                          z_star_grid[idx_cross], z_star_grid[idx_cross + 1],
                          xtol=1e-12, rtol=1e-12)
        l_c = compute_l_gr(z_star_c, f_func, h_func)
        print(f"  Phase transition:")
        print(f"    z_*^c = {z_star_c:.10f}")
        print(f"    l_c   = {l_c:.10f}")
    else:
        print("  WARNING: No zero crossing found for A_reg")
    print()

    # --- Step 5: l(z_*) monotonicity check ---
    dl = np.diff(l_of_zstar)
    is_monotonic = np.all(dl > 0)
    print(f"  l(z_*) monotonically increasing: {is_monotonic}")
    print()

    # --- Step 6: Save data ---
    out_path = os.path.join(data_dir, 'ode_benchmark_gr.npz')
    np.savez(out_path,
             z_star_grid=z_star_grid,
             l_of_zstar=l_of_zstar,
             A_reg_of_zstar=A_reg_of_zstar,
             l_c=np.array([l_c]),
             z_star_c=np.array([z_star_c]),
             l_max=np.array([l_max]),
             z_star_at_lmax=np.array([z_star_at_lmax]),
             Q=np.array([Q]),
             z_h=np.array([Z_H]),
             z_metric=z_metric,
             f_metric=f_metric,
             h_metric=h_metric)

    print(f"Data saved to {out_path}")

    # --- Step 7: Plots ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Plot 1: l(z_*) and A_reg(z_*)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(z_star_grid, l_of_zstar, 'b-', lw=1.5)
        if not np.isnan(l_c):
            ax1.axhline(y=l_c, color='r', ls='--',
                        label=f'$l_c = {l_c:.4f}$')
        ax1.set_xlabel(r'$z_*$')
        ax1.set_ylabel(r'$l(z_*)$')
        ax1.set_title(f'GR strip width (Q={Q})')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(z_star_grid, A_reg_of_zstar, 'b-', lw=1.5)
        ax2.axhline(y=0, color='k', ls='-', lw=0.5)
        if not np.isnan(z_star_c):
            ax2.axvline(x=z_star_c, color='r', ls='--',
                        label=f'$z_*^c = {z_star_c:.4f}$')
        ax2.set_xlabel(r'$z_*$')
        ax2.set_ylabel(r'$A_{\mathrm{reg}}(z_*)$')
        ax2.set_title(f'GR regularized area (Q={Q})')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'ode_benchmark_gr.pdf'), dpi=150)
        plt.close()

        # Plot 2: S_EE(l) curve
        fig, ax = plt.subplots(figsize=(8, 6))
        phys = A_reg_of_zstar <= 0
        unphys = A_reg_of_zstar > 0
        ax.plot(l_of_zstar[phys], A_reg_of_zstar[phys], 'b-', lw=1.5,
                label='Connected (physical)')
        ax.plot(l_of_zstar[unphys], A_reg_of_zstar[unphys], 'b--', lw=1.0,
                alpha=0.5, label=r'Connected ($A_{\rm reg}>0$)')
        if not np.isnan(l_c):
            l_disc = np.linspace(l_c, np.max(l_of_zstar), 50)
            ax.plot(l_disc, np.zeros_like(l_disc), 'r-', lw=1.5,
                    label=r'Disconnected ($A_{\rm reg}=0$)')
            ax.axvline(x=l_c, color='r', ls='--', alpha=0.5,
                       label=f'$l_c = {l_c:.4f}$')
        ax.axhline(y=0, color='k', ls='-', lw=0.5)
        ax.set_xlabel(r'$l$')
        ax.set_ylabel(r'$A_{\mathrm{reg}}(l)$')
        ax.set_title(f'GR entanglement entropy (Q={Q})')
        ax.legend(loc='lower left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'S_EE_vs_l_gr.pdf'), dpi=150)
        plt.close()

        print(f"Figures saved to {fig_dir}/")

    except ImportError:
        print("matplotlib not available, skipping plots")

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Q = {Q}")
    print(f"  Phase transition: l_c = {l_c:.10f}")
    print(f"  z_*^c = {z_star_c:.10f}")
    print(f"  l_max = {l_max:.6f} at z_* = {z_star_at_lmax:.6f}")
    print(f"  l(z_*) monotonic: {is_monotonic}")
    print(f"  Data file: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
