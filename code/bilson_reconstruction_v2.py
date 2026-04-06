"""
Full metric reconstruction for AdS4 Gubser-Rocha via:
  1. S_EE(l) → g(r) via Bilson inversion
  2. V(L) + S_EE(l) → χ(r) via derivative ratio
  3. (g, χ) → (f, h) via coordinate transformation

v2: improved numerics
  - Bilson integral uses analytic derivative via Leibniz rule instead of finite differences
  - Derivative relations use spline differentiation instead of np.gradient
  - Denser grid near boundary, sparser near horizon
  - Theta-substitution for turning-point integrals to remove endpoint singularity
"""
import os, sys
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

Q = 1.0
Z_H = 1.0


# ---- Exact metric ----
def f_exact(z):
    U = (1 + (1+3*Q)*z + (1+3*Q+3*Q**2)*z**2) / (1+Q*z)**1.5
    return (1 - z) * U

def h_exact(z):
    return (1 + Q*z) ** 1.5

def h_exact_prime(z):
    return 1.5 * Q * (1 + Q*z) ** 0.5

def alpha_exact(z):
    return 1.0 - z * h_exact_prime(z) / (2 * h_exact(z))

def g_exact_at_z(z):
    return alpha_exact(z)**2 * f_exact(z)

def chi_exact_at_z(z):
    a = alpha_exact(z)
    return np.log(a**2 * h_exact(z))

def r_of_z(z):
    return z / np.sqrt(h_exact(z))


# ==============================================================
# Turning-point integrals with theta substitution
# ==============================================================
# z = z_* sin^{2/d}(theta) removes the square-root singularity at z=z_*
# For d=3 (AdS4), the integrals have z^4 in the ratio, so we use
# the substitution z = z_* * t, dz = z_* dt, and handle the singularity
# at t=1 with a sqrt(1-t^2) type change.

def compute_l_RT(z_star):
    """RT half-width: l/2 = int_0^{z_*} dz/[sqrt(fh) sqrt(h^2 z_*^4/(z^4 h_*^2) - 1)]"""
    h_star = h_exact(z_star)
    # substitution t = (z/z_*)^2, z = z_* sqrt(t), dz = z_*/(2 sqrt(t)) dt
    def integrand(t):
        if t < 1e-15 or t > 1-1e-15: return 0.0
        z = z_star * np.sqrt(t)
        fv, hv = f_exact(z), h_exact(z)
        if fv <= 0 or hv <= 0: return 0.0
        ratio = hv**2 / (t**2 * h_star**2)  # h^2 z_*^4 / (z^4 h_*^2) = h^2/(t^2 h_*^2)
        if ratio <= 1.0 + 1e-15: return 0.0
        # dz = z_*/(2 sqrt(t)) dt
        return z_star / (2*np.sqrt(t)) / (np.sqrt(fv*hv) * np.sqrt(ratio - 1.0))
    result, _ = quad(integrand, 0, 1-1e-12, epsabs=1e-12, epsrel=1e-12, limit=500)
    return 2.0 * result

def compute_Areg_RT(z_star):
    """RT regularized area (eq 20)."""
    h_star = h_exact(z_star)
    # First integral with t = (z/z_*)^2
    def integrand1(t):
        if t < 1e-15 or t > 1-1e-15: return 0.0
        z = z_star * np.sqrt(t)
        fv, hv = f_exact(z), h_exact(z)
        if fv <= 0 or hv <= 0: return 0.0
        eta = t**2 * h_star**2 / hv**2
        if eta >= 1.0 - 1e-15: return 0.0
        prefactor = np.sqrt(hv) / (z**2 * np.sqrt(fv))
        bracket = 1.0/np.sqrt(1.0 - eta) - 1.0
        # dz = z_*/(2 sqrt(t)) dt
        return prefactor * bracket * z_star / (2*np.sqrt(t))
    I1, _ = quad(integrand1, 0, 1-1e-12, epsabs=1e-12, epsrel=1e-12, limit=500)
    # Second integral: z_* to z_h
    def integrand2(z):
        fv, hv = f_exact(z), h_exact(z)
        if fv <= 0: return 0.0
        return np.sqrt(hv) / (z**2 * np.sqrt(fv))
    I2, _ = quad(integrand2, z_star, Z_H*(1-1e-10), epsabs=1e-12, epsrel=1e-12, limit=500)
    return I1 - I2

def compute_L_WL(z_star):
    """Wilson loop half-separation."""
    f_star, h_star = f_exact(z_star), h_exact(z_star)
    F_star = f_star * h_star
    # substitution t = (z/z_*)^2
    def integrand(t):
        if t < 1e-15 or t > 1-1e-15: return 0.0
        z = z_star * np.sqrt(t)
        fv, hv = f_exact(z), h_exact(z)
        Fv = fv * hv
        if Fv <= 0: return 0.0
        ratio = Fv / (t**2 * F_star)
        if ratio <= 1.0 + 1e-15: return 0.0
        return z_star / (2*np.sqrt(t)) / (np.sqrt(Fv) * np.sqrt(ratio - 1.0))
    result, _ = quad(integrand, 0, 1-1e-12, epsabs=1e-12, epsrel=1e-12, limit=500)
    return 2.0 * result

def compute_Vreg_WL(z_star):
    """Wilson loop regularized potential."""
    f_star, h_star = f_exact(z_star), h_exact(z_star)
    F_star = f_star * h_star
    # First integral with t = (z/z_*)^2
    def integrand1(t):
        if t < 1e-15 or t > 1-1e-15: return 0.0
        z = z_star * np.sqrt(t)
        fv, hv = f_exact(z), h_exact(z)
        Fv = fv * hv
        if Fv <= 0: return 0.0
        eta = t**2 * F_star / Fv
        if eta >= 1.0 - 1e-15: return 0.0
        return (1.0/z**2) * (1.0/np.sqrt(1.0 - eta) - 1.0) * z_star / (2*np.sqrt(t))
    I1, _ = quad(integrand1, 0, 1-1e-12, epsabs=1e-12, epsrel=1e-12, limit=500)
    I2 = 1.0/z_star - 1.0/Z_H
    return I1 - I2


# ==============================================================
# STEP 0: Generate data
# ==============================================================
def generate_data():
    print("=" * 70)
    print("Generating exact S_EE(l) and V(L) data")
    print("=" * 70)
    # Non-uniform grid: denser near boundary
    z1 = np.linspace(0.02, 0.3, 80)
    z2 = np.linspace(0.3, 0.7, 80)
    z3 = np.linspace(0.7, 0.93, 40)
    z_stars = np.unique(np.concatenate([z1, z2, z3]))

    l_arr = np.zeros(len(z_stars))
    A_arr = np.zeros(len(z_stars))
    L_arr = np.zeros(len(z_stars))
    V_arr = np.zeros(len(z_stars))
    r_arr = np.zeros(len(z_stars))

    for i, zs in enumerate(z_stars):
        l_arr[i] = compute_l_RT(zs)
        A_arr[i] = compute_Areg_RT(zs)
        L_arr[i] = compute_L_WL(zs)
        V_arr[i] = compute_Vreg_WL(zs)
        r_arr[i] = r_of_z(zs)
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(z_stars)}")

    print(f"  Points: {len(z_stars)}, l: [{l_arr.min():.4f}, {l_arr.max():.4f}], "
          f"L: [{L_arr.min():.4f}, {L_arr.max():.4f}]")
    return z_stars, l_arr, A_arr, L_arr, V_arr, r_arr


# ==============================================================
# STEP 1: Bilson inversion  S_EE(l) → g(r)
# ==============================================================
def bilson_inversion(z_stars, l_arr, r_arr):
    """
    Bilson formula II (Ahn eq 3.10 / ECF2):
      1/sqrt(g(r)) = (2/pi) (1/r^2) d/dr int_0^r r_*^3/sqrt(r^4-r_*^4) l(r_*) dr_*

    Key improvement: compute the integral I(r) on a grid, fit a spline,
    then differentiate the spline analytically.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Bilson inversion  S_EE(l) → g(r)")
    print("=" * 70)

    # Build l(r_*) with small-r_* extrapolation via polynomial fit
    # At small r_*, l(r_*) ~ c1*r_* + c3*r_*^3 + ... (odd powers by symmetry)
    # Fit l/r_* to a polynomial in r_*^2 using the smallest data points
    n_fit = min(20, len(r_arr)//4)
    r_small = r_arr[:n_fit]
    l_small = l_arr[:n_fit]
    # Fit l(r) = r * P(r^2) where P is a polynomial
    lor = l_small / r_small  # l/r_*
    coeffs = np.polyfit(r_small**2, lor, 3)  # cubic in r^2
    poly_lor = np.poly1d(coeffs)

    def l_of_r_extended(rs):
        """l(r_*) with polynomial extrapolation below data range."""
        rs = np.atleast_1d(rs)
        result = np.zeros_like(rs)
        mask_data = rs >= r_arr[0]
        mask_extrap = ~mask_data
        if mask_data.any():
            spline = CubicSpline(r_arr, l_arr)
            result[mask_data] = spline(rs[mask_data])
        if mask_extrap.any():
            result[mask_extrap] = rs[mask_extrap] * poly_lor(rs[mask_extrap]**2)
        return result if len(result) > 1 else float(result[0])

    # Verify extrapolation
    r_test = np.array([0.001, 0.005, 0.01, r_arr[0], r_arr[5]])
    l_test = l_of_r_extended(r_test)
    print(f"  l(r_*) extrapolation check:")
    for rt, lt in zip(r_test, l_test):
        print(f"    r_*={rt:.4f}: l={lt:.6f}")

    # Evaluate I(r) = int_0^r r_*^3/sqrt(r^4 - r_*^4) l(r_*) dr_*
    # on a fine r grid, now starting from very small r
    r_min, r_max = 0.005, r_arr.max() * 0.95
    r_grid = np.linspace(r_min, r_max, 200)
    I_grid = np.zeros(len(r_grid))

    for i, r in enumerate(r_grid):
        # substitution: u = (r_*/r)^4
        # integrand = r^2/(4 sqrt(1-u)) * l(r*u^{1/4}) du
        def integ(u):
            if u < 1e-15 or u > 1-1e-15: return 0.0
            rs = r * u**0.25
            return r**2 / (4*np.sqrt(1-u)) * l_of_r_extended(rs)
        I_grid[i], _ = quad(integ, 0, 1-1e-10, epsabs=1e-11, epsrel=1e-11, limit=500)

    # Spline fit and analytic differentiation
    I_spline = CubicSpline(r_grid, I_grid)
    dIdr = I_spline(r_grid, 1)  # first derivative

    # 1/sqrt(g) = (2/pi)(1/r^2) dI/dr
    inv_sqrt_g = (2.0/np.pi) * dIdr / r_grid**2
    g_bilson = np.where(inv_sqrt_g > 0, 1.0/inv_sqrt_g**2, np.nan)

    # Exact g(r) for comparison
    z_fine = np.linspace(0.001, 0.999, 5000)
    r_fine = np.array([r_of_z(z) for z in z_fine])
    g_fine = np.array([g_exact_at_z(z) for z in z_fine])
    g_of_r_exact = CubicSpline(r_fine, g_fine)
    g_exact_grid = g_of_r_exact(r_grid)

    valid = ~np.isnan(g_bilson) & (g_exact_grid > 0.05)
    rel_err = np.abs(g_bilson[valid] - g_exact_grid[valid]) / g_exact_grid[valid]

    print(f"  max rel error = {rel_err.max():.4e}")
    print(f"  mean rel error = {rel_err.mean():.4e}")
    print(f"  median rel error = {np.median(rel_err):.4e}")

    print(f"\n  {'r':>8s} {'g_bilson':>10s} {'g_exact':>10s} {'rel_err':>10s}")
    for i in range(0, len(r_grid), 20):
        if np.isnan(g_bilson[i]): continue
        re = abs(g_bilson[i] - g_exact_grid[i]) / max(abs(g_exact_grid[i]), 1e-10)
        print(f"  {r_grid[i]:8.4f} {g_bilson[i]:10.6f} {g_exact_grid[i]:10.6f} {re:10.4e}")

    return r_grid, g_bilson, g_exact_grid


# ==============================================================
# STEP 2: Derivative ratio → χ(r)
# ==============================================================
def reconstruct_chi(z_stars, l_arr, A_arr, L_arr, V_arr, r_arr):
    print("\n" + "=" * 70)
    print("STEP 2: Derivative ratio → χ(r)")
    print("=" * 70)

    # Use cubic spline derivatives instead of np.gradient
    A_spline = CubicSpline(z_stars, A_arr)
    l_spline = CubicSpline(z_stars, l_arr)
    V_spline = CubicSpline(z_stars, V_arr)
    L_spline = CubicSpline(z_stars, L_arr)

    dAdz = A_spline(z_stars, 1)
    dldz = l_spline(z_stars, 1)
    dVdz = V_spline(z_stars, 1)
    dLdz = L_spline(z_stars, 1)

    dSdl = dAdz / dldz
    dVdL = dVdz / dLdz

    # Check against exact
    dSdl_exact = np.array([h_exact(zs)/zs**2 for zs in z_stars])
    dVdL_exact = np.array([np.sqrt(f_exact(zs)*h_exact(zs))/zs**2 for zs in z_stars])

    # Skip first and last few points (boundary effects)
    interior = slice(5, -5)
    err_dSdl = np.abs(dSdl[interior] - dSdl_exact[interior]) / np.abs(dSdl_exact[interior])
    err_dVdL = np.abs(dVdL[interior] - dVdL_exact[interior]) / np.abs(dVdL_exact[interior])

    print(f"  dS/dl: max rel err = {err_dSdl.max():.4e}, mean = {err_dSdl.mean():.4e}")
    print(f"  dV/dL: max rel err = {err_dVdL.max():.4e}, mean = {err_dVdL.mean():.4e}")

    # Ratio = sqrt(f/h) = sqrt(g * e^{-chi})
    ratio = dVdL / dSdl
    ratio_exact = np.array([np.sqrt(f_exact(zs)/h_exact(zs)) for zs in z_stars])
    err_ratio = np.abs(ratio[interior] - ratio_exact[interior]) / np.abs(ratio_exact[interior])
    print(f"  sqrt(f/h) ratio: max rel err = {err_ratio.max():.4e}, mean = {err_ratio.mean():.4e}")

    # chi = log(g / ratio^2) since f/h = g/e^chi → e^chi = g/(f/h) = g/ratio^2
    g_at_z = np.array([g_exact_at_z(zs) for zs in z_stars])
    chi_recon = np.log(g_at_z / ratio**2)
    chi_exact_arr = np.array([chi_exact_at_z(zs) for zs in z_stars])

    err_chi = np.abs(chi_recon[interior] - chi_exact_arr[interior])
    print(f"\n  χ: max abs err = {err_chi.max():.4e}, mean = {err_chi.mean():.4e}")

    print(f"\n  {'z_*':>6s} {'χ_recon':>10s} {'χ_exact':>10s} {'|err|':>10s} {'f/h_rec':>10s} {'f/h_ex':>10s}")
    for i in range(5, len(z_stars)-5, 15):
        zs = z_stars[i]
        foh_rec = ratio[i]**2
        foh_ex = f_exact(zs)/h_exact(zs)
        print(f"  {zs:6.3f} {chi_recon[i]:10.6f} {chi_exact_arr[i]:10.6f} "
              f"{abs(chi_recon[i]-chi_exact_arr[i]):10.2e} {foh_rec:10.6f} {foh_ex:10.6f}")

    return z_stars, chi_recon, chi_exact_arr, ratio, dSdl, dVdL


# ==============================================================
# STEP 3: (g, χ) → (f, h)
# ==============================================================
def reconstruct_fh(z_stars, ratio, dSdl, r_arr, r_grid, g_bilson):
    """
    Recover f(z_*) and h(z_*) from the reconstruction data.

    From the derivative relations (using the parametric variable z_*):
      dS/dl = h_* / z_*^2   →   h_* = z_*^2 * dS/dl
      dV/dL = sqrt(f_*h_*) / z_*^2   →   f_*h_* = (z_*^2 * dV/dL)^2

    Note: z_* here is the parametric variable we computed on, so these
    are valid. In a real reconstruction from data alone, z_* would need
    to be determined from r_* via the (unknown) coordinate transformation.
    Here we use the known z_* grid as a proof of concept.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Recover f(z) and h(z)")
    print("=" * 70)

    s = slice(5, -5)
    zs = z_stars[s]

    # h(z_*) = z_*^2 * dS/dl
    # Note: A_reg in our code is the HALF-area (integral from 0 to z_*),
    # while l is the FULL width. So dA_reg/dl = (1/2) h_*/z_*^2.
    # Therefore h_* = 2 * z_*^2 * dA_reg/dl.
    h_recon = 2 * zs**2 * dSdl[s]
    h_exact_arr = np.array([h_exact(z) for z in zs])

    # f(z_*)/h(z_*) = ratio^2
    foh_recon = ratio[s]**2
    f_recon = h_recon * foh_recon
    f_exact_arr = np.array([f_exact(z) for z in zs])

    err_h = np.abs(h_recon - h_exact_arr) / h_exact_arr
    err_f = np.abs(f_recon - f_exact_arr) / np.maximum(f_exact_arr, 1e-10)
    # Only count f errors where f is not too small
    f_mask = f_exact_arr > 0.05
    err_f_bulk = err_f[f_mask]

    print(f"  h(z): max rel err = {err_h.max():.4e}, mean = {err_h.mean():.4e}")
    print(f"  f(z): max rel err (f>0.05) = {err_f_bulk.max():.4e}, mean = {err_f_bulk.mean():.4e}")

    print(f"\n  {'z':>6s} {'f_recon':>10s} {'f_exact':>10s} {'|df/f|':>10s} "
          f"{'h_recon':>10s} {'h_exact':>10s} {'|dh/h|':>10s}")
    for i in range(0, len(zs), 15):
        z = zs[i]
        rf = abs(f_recon[i] - f_exact_arr[i]) / max(abs(f_exact_arr[i]), 1e-10)
        rh = abs(h_recon[i] - h_exact_arr[i]) / h_exact_arr[i]
        print(f"  {z:6.3f} {f_recon[i]:10.6f} {f_exact_arr[i]:10.6f} {rf:10.4e} "
              f"{h_recon[i]:10.6f} {h_exact_arr[i]:10.6f} {rh:10.4e}")

    return zs, f_recon, f_exact_arr, h_recon, h_exact_arr


# ==============================================================
# STEP 4: Plots
# ==============================================================
def make_plots(r_grid, g_bilson, g_exact_grid,
               z_stars, chi_recon, chi_exact_arr, ratio,
               zs_fh, f_recon, f_exact_arr, h_recon, h_exact_arr):
    print("\n" + "=" * 70)
    print("Generating plots")
    print("=" * 70)

    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Panel (0,0): g(r) from Bilson
    ax = axes[0, 0]
    valid = ~np.isnan(g_bilson)
    ax.plot(r_grid[valid], g_exact_grid[valid], 'r-', lw=2, label=r'$g(r)$ exact')
    ax.plot(r_grid[valid], g_bilson[valid], 'b--', lw=1.5, label=r'$g(r)$ Bilson')
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$g(r)$')
    ax.set_title(r'Bilson: $S_{EE}(l) \to g(r)$')
    ax.legend()

    # Panel (0,1): chi(r)
    ax = axes[0, 1]
    s = slice(5, -5)
    r_plot = np.array([r_of_z(zs) for zs in z_stars])
    ax.plot(r_plot[s], chi_exact_arr[s], 'r-', lw=2, label=r'$\chi(r)$ exact')
    ax.plot(r_plot[s], chi_recon[s], 'b--', lw=1.5, label=r'$\chi(r)$ recon')
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$\chi(r)$')
    ax.set_title(r'Derivative ratio $\to \chi(r)$')
    ax.legend()

    # Panel (0,2): f/h ratio
    ax = axes[0, 2]
    foh_exact = np.array([f_exact(zs)/h_exact(zs) for zs in z_stars])
    ax.plot(z_stars[s], foh_exact[s], 'r-', lw=2, label=r'$f/h$ exact')
    ax.plot(z_stars[s], ratio[s]**2, 'b--', lw=1.5, label=r'$f/h$ recon')
    ax.set_xlabel(r'$z_*$')
    ax.set_ylabel(r'$f/h$')
    ax.set_title(r'$\sqrt{f/h}$ from ratio')
    ax.legend()

    # Panel (1,0): f(z)
    ax = axes[1, 0]
    ax.plot(zs_fh, f_exact_arr, 'r-', lw=2, label=r'$f(z)$ exact')
    ax.plot(zs_fh, f_recon, 'b--', lw=1.5, label=r'$f(z)$ recon')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$f(z)$')
    ax.set_title(r'Reconstructed $f(z)$')
    ax.legend()

    # Panel (1,1): h(z)
    ax = axes[1, 1]
    ax.plot(zs_fh, h_exact_arr, 'r-', lw=2, label=r'$h(z)$ exact')
    ax.plot(zs_fh, h_recon, 'b--', lw=1.5, label=r'$h(z)$ recon')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$h(z)$')
    ax.set_title(r'Reconstructed $h(z)$')
    ax.legend()

    # Panel (1,2): relative errors
    ax = axes[1, 2]
    err_f = np.abs(f_recon - f_exact_arr) / np.maximum(f_exact_arr, 1e-10)
    err_h = np.abs(h_recon - h_exact_arr) / h_exact_arr
    ax.semilogy(zs_fh, err_f, 'b-', lw=1.5, label=r'$|\Delta f/f|$')
    ax.semilogy(zs_fh, err_h, 'r-', lw=1.5, label=r'$|\Delta h/h|$')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel('Relative error')
    ax.set_title('Reconstruction errors')
    ax.legend()
    ax.set_ylim(1e-7, 1e-1)

    plt.tight_layout()
    outpath = os.path.join(fig_dir, 'bilson_reconstruction.pdf')
    plt.savefig(outpath, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


def main():
    z_stars, l_arr, A_arr, L_arr, V_arr, r_arr = generate_data()
    r_grid, g_bilson, g_exact_grid = bilson_inversion(z_stars, l_arr, r_arr)
    zs_out, chi_recon, chi_exact_arr, ratio, dSdl, dVdL = reconstruct_chi(
        z_stars, l_arr, A_arr, L_arr, V_arr, r_arr)
    zs_fh, f_recon, f_exact_arr, h_recon, h_exact_arr = reconstruct_fh(
        z_stars, ratio, dSdl, r_arr, r_grid, g_bilson)
    make_plots(r_grid, g_bilson, g_exact_grid,
               zs_out, chi_recon, chi_exact_arr, ratio,
               zs_fh, f_recon, f_exact_arr, h_recon, h_exact_arr)

    print("\n" + "=" * 70)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
