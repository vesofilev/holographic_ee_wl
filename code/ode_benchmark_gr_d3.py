# ASSERT_CONVENTION: AdS4 (d=3 boundary) Gubser-Rocha
# ASSERT_CONVENTION: area = Omega int (sqrt(h)/z^2) sqrt(h + z'^2/f) dx
# ASSERT_CONVENTION: disc = Omega int sqrt(h)/(z^2 sqrt(f)) dz
"""
ODE Benchmark for Strip Entanglement Entropy in AdS4 Gubser-Rocha.

Closed-form metric: f(z) = (1-z)U(z), h(z) = (1+Qz)^{3/2}.

Strip half-width (d=3):
  l/2 = int_0^{z_*} dz / [sqrt(f h) sqrt(h^2 z_*^4 / (z^4 h_*^2) - 1)]

Regularized area (d=3):
  A_reg = int_0^{z_*} (sqrt(h)/z^2)/sqrt(f) [1/sqrt(1 - z^4 h_*^2/(z_*^4 h^2)) - 1] dz
        - int_{z_*}^{1} sqrt(h)/(z^2 sqrt(f)) dz
"""

import os
import sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gubser_rocha_d3 import gr_metric_d3, gr_h_horizon, gr_temperature


Z_H = 1.0
N_ZSTAR = 500


def f_exact(z, Q):
    """Closed-form f(z) = (1-z)U(z)."""
    U = (1 + (1+3*Q)*z + (1+3*Q+3*Q**2)*z**2) / (1+Q*z)**1.5
    return (1 - z) * U


def h_exact(z, Q):
    """Closed-form h(z) = (1+Qz)^{3/2}."""
    return (1 + Q*z) ** 1.5


def compute_l_d3(z_star, Q):
    """Strip width l for AdS4 GR (d=3)."""
    h_star = h_exact(z_star, Q)

    def integrand(z):
        if z < 1e-15:
            return 0.0
        fv = f_exact(z, Q)
        hv = h_exact(z, Q)
        if fv <= 0 or hv <= 0:
            return 0.0
        ratio = hv**2 * z_star**4 / (z**4 * h_star**2)
        if ratio <= 1.0 + 1e-15:
            return 0.0
        return 1.0 / (np.sqrt(fv * hv) * np.sqrt(ratio - 1.0))

    result, _ = quad(integrand, 0, z_star * (1.0 - 1e-12),
                     epsabs=1e-12, epsrel=1e-12, limit=400,
                     points=[z_star * 0.5])
    return 2.0 * result


def compute_A_reg_d3(z_star, Q, z_h=Z_H):
    """Regularized area for AdS4 GR (d=3)."""
    h_star = h_exact(z_star, Q)

    # First integral: 0 to z_*
    def integrand1(z):
        if z < 1e-15:
            return 0.0
        fv = f_exact(z, Q)
        hv = h_exact(z, Q)
        if fv <= 0 or hv <= 0:
            return 0.0
        ratio4 = z**4 * h_star**2 / (z_star**4 * hv**2)
        if ratio4 >= 1.0 - 1e-15:
            return 0.0
        return (np.sqrt(hv) / (z**2 * np.sqrt(fv))) * (
            1.0 / np.sqrt(1.0 - ratio4) - 1.0)

    I1, _ = quad(integrand1, 0, z_star * (1.0 - 1e-12),
                 epsabs=1e-12, epsrel=1e-12, limit=400,
                 points=[z_star * 0.5])

    # Second integral: z_* to z_h
    def integrand2(z):
        fv = f_exact(z, Q)
        hv = h_exact(z, Q)
        if fv <= 0:
            return 0.0
        return np.sqrt(hv) / (z**2 * np.sqrt(fv))

    I2, _ = quad(integrand2, z_star, z_h * (1.0 - 1e-12),
                 epsabs=1e-12, epsrel=1e-12, limit=400,
                 points=[z_star + (z_h - z_star) * 0.5])

    return I1 - I2


def validate_Q0():
    """Q=0: should match AdS4-Schwarzschild f=1-z^3, h=1."""
    print("Q=0 validation (AdS4-Schwarzschild):")
    # For f=1-z^3, h=1 (d=3): known benchmark
    for zs in [0.3, 0.5, 0.7]:
        l = compute_l_d3(zs, Q=0.0)
        A = compute_A_reg_d3(zs, Q=0.0)
        print(f"  z*={zs:.1f}: l={l:.6f}, A_reg={A:.6f}")
    print()


def main():
    Q = 1.0
    print("=" * 60)
    print(f"ODE Benchmark: AdS4 Gubser-Rocha (Q={Q})")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    validate_Q0()

    # Metric
    z_metric, f_metric, h_metric = gr_metric_d3(Q, n_points=4000)
    h_h = gr_h_horizon(Q)
    T = gr_temperature(Q)
    print(f"  h(0)={h_exact(0,Q):.6f}, h(0.5)={h_exact(0.5,Q):.6f}, "
          f"h(1)={h_h:.6f}")
    print(f"  T={T:.6f}")
    print()

    # Compute l(z_*) and A_reg(z_*)
    z_star_low = np.linspace(0.01, 0.60, 150)
    z_star_mid = np.linspace(0.60, 0.95, 250)
    z_star_high = np.linspace(0.95, 0.99, 100)
    z_star_grid = np.unique(np.concatenate([z_star_low, z_star_mid, z_star_high]))

    print(f"Computing l(z_*) and A_reg(z_*) for {len(z_star_grid)} points...")
    l_of_zstar = np.zeros(len(z_star_grid))
    A_reg_of_zstar = np.zeros(len(z_star_grid))

    for i, zs in enumerate(z_star_grid):
        l_of_zstar[i] = compute_l_d3(zs, Q)
        A_reg_of_zstar[i] = compute_A_reg_d3(zs, Q)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(z_star_grid)}")

    print(f"  l range: [{l_of_zstar.min():.6f}, {l_of_zstar.max():.6f}]")
    print(f"  A_reg range: [{A_reg_of_zstar.min():.6f}, {A_reg_of_zstar.max():.6f}]")

    # Phase transition: A_reg = 0
    signs = np.sign(A_reg_of_zstar)
    sign_changes = np.where(np.diff(signs) != 0)[0]
    l_c = np.nan
    if len(sign_changes) > 0:
        idx = sign_changes[-1]
        z_star_c = brentq(lambda zs: compute_A_reg_d3(zs, Q),
                          z_star_grid[idx], z_star_grid[idx+1],
                          xtol=1e-12)
        l_c = compute_l_d3(z_star_c, Q)
        print(f"  Phase transition: l_c = {l_c:.10f}, z*_c = {z_star_c:.10f}")
    else:
        z_star_c = np.nan
        print("  No phase transition found")

    # Save
    out_path = os.path.join(data_dir, 'ode_benchmark_gr_d3.npz')
    np.savez(out_path,
             z_star_grid=z_star_grid,
             l_of_zstar=l_of_zstar,
             A_reg_of_zstar=A_reg_of_zstar,
             l_c=np.array([l_c]),
             z_star_c=np.array([z_star_c if not np.isnan(z_star_c) else 0.0]),
             Q=np.array([Q]),
             z_h=np.array([Z_H]),
             z_metric=z_metric,
             f_metric=f_metric,
             h_metric=h_metric)
    print(f"\nSaved: {out_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(z_star_grid, l_of_zstar, 'b-', lw=1.5)
        if not np.isnan(l_c):
            ax1.axhline(y=l_c, color='r', ls='--', label=f'$l_c={l_c:.4f}$')
        ax1.set_xlabel(r'$z_*$'); ax1.set_ylabel(r'$l(z_*)$')
        ax1.set_title(f'AdS$_4$ GR strip width (Q={Q})')
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(z_star_grid, A_reg_of_zstar, 'b-', lw=1.5)
        ax2.axhline(y=0, color='k', ls='-', lw=0.5)
        ax2.set_xlabel(r'$z_*$'); ax2.set_ylabel(r'$A_{\mathrm{reg}}(z_*)$')
        ax2.set_title(f'AdS$_4$ GR regularized area (Q={Q})')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'ode_benchmark_gr_d3.pdf'), dpi=150)
        plt.close()
        print(f"Saved figure")
    except ImportError:
        pass

    print("\nDONE.")


if __name__ == "__main__":
    main()
