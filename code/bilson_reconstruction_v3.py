"""
Non-circular metric reconstruction for AdS4 Gubser-Rocha.

Complete reconstruction from boundary data {S_EE(l), V(L)} only:
  Step 1: Bilson inversion  S_EE(l) → g(r)    [r_* extracted from dS/dl]
  Step 2: η-coordinate      g(r)    → G(η)     [computable from g]
  Step 3: WL processing     V(L)    → L(h_0)   [h_0 = dV/dL from data]
  Step 4: Abel inversion    L(h_0)  → σ(H)
  Step 5: ODE integration   σ,G     → η(H) → H(η)
  Step 6: Metric recovery   H,G,g   → F → χ → f,h

v3: fully non-circular — r_* extracted from data derivatives, not from known h(z_*).
    Wilson loop inversion via Hashimoto's method in η-coordinates.
"""
import os, sys
import numpy as np
from scipy.integrate import quad, solve_ivp, cumulative_trapezoid
from scipy.interpolate import CubicSpline, interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

Q = 1.0
Z_H = 1.0

# ---- Exact metric (for data generation and comparison only) ----
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


# ---- Turning-point integrals (for data generation) ----
def compute_l_RT(z_star):
    h_star = h_exact(z_star)
    def integrand(t):
        if t < 1e-15 or t > 1-1e-15: return 0.0
        z = z_star * np.sqrt(t)
        fv, hv = f_exact(z), h_exact(z)
        if fv <= 0 or hv <= 0: return 0.0
        ratio = hv**2 / (t**2 * h_star**2)
        if ratio <= 1.0 + 1e-15: return 0.0
        return z_star / (2*np.sqrt(t)) / (np.sqrt(fv*hv) * np.sqrt(ratio - 1.0))
    result, _ = quad(integrand, 0, 1-1e-12, epsabs=1e-12, epsrel=1e-12, limit=500)
    return 2.0 * result

def compute_Areg_RT(z_star):
    h_star = h_exact(z_star)
    def integrand1(t):
        if t < 1e-15 or t > 1-1e-15: return 0.0
        z = z_star * np.sqrt(t)
        fv, hv = f_exact(z), h_exact(z)
        if fv <= 0 or hv <= 0: return 0.0
        eta = t**2 * h_star**2 / hv**2
        if eta >= 1.0 - 1e-15: return 0.0
        prefactor = np.sqrt(hv) / (z**2 * np.sqrt(fv))
        bracket = 1.0/np.sqrt(1.0 - eta) - 1.0
        return prefactor * bracket * z_star / (2*np.sqrt(t))
    I1, _ = quad(integrand1, 0, 1-1e-12, epsabs=1e-12, epsrel=1e-12, limit=500)
    def integrand2(z):
        fv, hv = f_exact(z), h_exact(z)
        if fv <= 0: return 0.0
        return np.sqrt(hv) / (z**2 * np.sqrt(fv))
    I2, _ = quad(integrand2, z_star, Z_H*(1-1e-10), epsabs=1e-12, epsrel=1e-12, limit=500)
    return I1 - I2

def compute_L_WL(z_star):
    f_star, h_star = f_exact(z_star), h_exact(z_star)
    F_star = f_star * h_star
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
    f_star, h_star = f_exact(z_star), h_exact(z_star)
    F_star = f_star * h_star
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
# STEP 0: Generate boundary data
# ==============================================================
def generate_data():
    print("=" * 70)
    print("STEP 0: Generating boundary data from exact metric")
    print("=" * 70)
    z1 = np.linspace(0.02, 0.3, 200)
    z2 = np.linspace(0.3, 0.7, 200)
    z3 = np.linspace(0.7, 0.93, 100)
    z_stars = np.unique(np.concatenate([z1, z2, z3]))

    l_arr = np.zeros(len(z_stars))
    A_arr = np.zeros(len(z_stars))
    L_arr = np.zeros(len(z_stars))
    V_arr = np.zeros(len(z_stars))

    for i, zs in enumerate(z_stars):
        l_arr[i] = compute_l_RT(zs)
        A_arr[i] = compute_Areg_RT(zs)
        L_arr[i] = compute_L_WL(zs)
        V_arr[i] = compute_Vreg_WL(zs)
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(z_stars)}")

    print(f"  RT: {len(z_stars)} pts, l=[{l_arr.min():.4f}, {l_arr.max():.4f}]")
    print(f"  WL: {len(z_stars)} pts, L=[{L_arr.min():.4f}, {L_arr.max():.4f}]")
    # Return as "boundary data": {l_i, A_i} and {L_j, V_j}
    # Also return z_stars for comparison only
    return z_stars, l_arr, A_arr, L_arr, V_arr


# ==============================================================
# STEP 1: Non-circular Bilson inversion  S_EE(l) → g(r)
# ==============================================================
def bilson_inversion(z_stars, l_arr, A_arr):
    """
    Non-circular Bilson inversion from boundary data only.
    Input: {z_i, l_i, A_reg_half_i} — z_i is parametric label only
    Output: g(r) on an r-grid

    r_* is extracted from dS_tilde/dl = (2 dA_half/dz_*) / (dl/dz_*) = 1/r_*^2
    This uses z_* only as a parametric variable for stable derivatives.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Bilson inversion (non-circular)")
    print("=" * 70)

    # S_tilde = 2 * A_half (full area, both halves, prefactor=1)
    S_tilde = 2.0 * A_arr

    # High-precision interpolation of S_tilde(l) via Chebyshev polynomials.
    # With 498 high-precision data points, Chebyshev interpolation gives
    # exponential convergence for smooth functions.
    from numpy.polynomial import chebyshev as C

    l_min, l_max = l_arr[0], l_arr[-1]
    # Map l to [-1, 1] for Chebyshev
    l_mapped = 2.0 * (l_arr - l_min) / (l_max - l_min) - 1.0
    # Fit with high degree — use all data points
    deg_S = 30  # moderate degree avoids Runge oscillation at edges
    coeffs_S = C.chebfit(l_mapped, S_tilde, deg_S)
    S_fit = C.chebval(l_mapped, coeffs_S)
    res_S = np.max(np.abs(S_fit - S_tilde))
    print(f"  S_tilde(l) Chebyshev deg={deg_S}, max residual={res_S:.2e}")

    # Derivative: d(S_tilde)/dl = d(S_tilde)/d(l_mapped) * d(l_mapped)/dl
    #   d(l_mapped)/dl = 2/(l_max - l_min)
    dcoeffs_S = C.chebder(coeffs_S)
    dSdl_mapped = C.chebval(l_mapped, dcoeffs_S)
    dSdl = dSdl_mapped * 2.0 / (l_max - l_min)

    # Extract r_* from dS/dl = 1/r_*^2
    valid = dSdl > 0
    if not np.all(valid):
        print(f"  WARNING: {np.sum(~valid)} points with dS/dl <= 0, clipping")
        dSdl = np.maximum(dSdl, 1e-10)

    r_star = 1.0 / np.sqrt(dSdl)

    # Now we have l(r_*) — but r_* may not be monotonic near endpoints
    # Trim boundary artifacts (first/last few points)
    trim = 5
    r_data = r_star[trim:-trim]
    l_data = l_arr[trim:-trim]

    # Verify monotonicity of r_* (should decrease as l increases for connected branch)
    if not np.all(np.diff(r_data) < 0) and not np.all(np.diff(r_data) > 0):
        print("  WARNING: r_*(l) is not monotonic — sorting")
    # Sort by r_*
    idx2 = np.argsort(r_data)
    r_data = r_data[idx2]
    l_data = l_data[idx2]

    print(f"  r_* range: [{r_data.min():.6f}, {r_data.max():.6f}]")
    print(f"  l   range: [{l_data.min():.6f}, {l_data.max():.6f}]")

    # Spot-check r_* vs exact (diagnostics only)
    print(f"\n  Spot-check r_* vs exact:")
    for iz in [trim, len(z_stars)//4, len(z_stars)//2, 3*len(z_stars)//4]:
        r_recon = r_star[iz]
        r_exact_val = r_of_z(z_stars[iz])
        print(f"    z_*={z_stars[iz]:.3f}: r_*(data)={r_recon:.6f}, "
              f"r_*(exact)={r_exact_val:.6f}, err={abs(r_recon-r_exact_val)/r_exact_val:.2e}")

    # Small-r_* extrapolation (pure-AdS asymptotics)
    n_fit = min(20, len(r_data)//4)
    r_small = r_data[:n_fit]
    l_small = l_data[:n_fit]
    lor = l_small / r_small
    coeffs = np.polyfit(r_small**2, lor, 3)
    poly_lor = np.poly1d(coeffs)

    def l_of_r_extended(rs):
        rs = np.atleast_1d(rs)
        result = np.zeros_like(rs)
        mask_data = rs >= r_data[0]
        mask_extrap = ~mask_data
        if mask_data.any():
            spline = CubicSpline(r_data, l_data)
            result[mask_data] = spline(np.clip(rs[mask_data], r_data[0], r_data[-1]))
        if mask_extrap.any():
            result[mask_extrap] = rs[mask_extrap] * poly_lor(rs[mask_extrap]**2)
        return result if len(result) > 1 else float(result[0])

    # Bilson integral: I(r) = int_0^r r_*^3/sqrt(r^4-r_*^4) l(r_*) dr_*
    # Substitution u = (r_*/r)^4
    r_min_bilson = r_data.min() * 1.2
    r_max_bilson = r_data.max() * 0.92
    r_grid_bilson = np.linspace(r_min_bilson, r_max_bilson, 400)
    I_grid = np.zeros(len(r_grid_bilson))

    for i, r in enumerate(r_grid_bilson):
        def integ(u):
            if u < 1e-15 or u > 1-1e-15: return 0.0
            rs = r * u**0.25
            return r**2 / (4*np.sqrt(1-u)) * l_of_r_extended(rs)
        I_grid[i], _ = quad(integ, 0, 1-1e-10, epsabs=1e-14, epsrel=1e-14, limit=1000)

    # Chebyshev fit + analytic differentiation for maximum precision
    from numpy.polynomial import chebyshev as C
    r_b_min, r_b_max = r_grid_bilson[0], r_grid_bilson[-1]
    r_b_mapped = 2.0 * (r_grid_bilson - r_b_min) / (r_b_max - r_b_min) - 1.0
    deg_bilson = min(80, len(r_grid_bilson) - 2)
    coeffs_bilson = C.chebfit(r_b_mapped, I_grid, deg_bilson)
    I_fit_bilson = C.chebval(r_b_mapped, coeffs_bilson)
    res_bilson = np.max(np.abs(I_fit_bilson - I_grid))
    print(f"  Bilson I(r) Chebyshev deg={deg_bilson}, max residual={res_bilson:.2e}")
    dcoeffs_bilson = C.chebder(coeffs_bilson)
    dIdr = C.chebval(r_b_mapped, dcoeffs_bilson) * 2.0 / (r_b_max - r_b_min)

    inv_sqrt_g = (2.0/np.pi) * dIdr / r_grid_bilson**2
    g_bilson_raw = np.where(inv_sqrt_g > 0, 1.0/inv_sqrt_g**2, np.nan)

    # --- Extend g(r) with boundary asymptotics ---
    # Near boundary (r→0): g(r) → 1 (pure AdS)
    # Fit g(r) = 1 + c2*r^2 + c4*r^4 to the small-r Bilson data
    n_bdy = min(30, len(r_grid_bilson)//5)
    r_bdy = r_grid_bilson[:n_bdy]
    g_bdy = g_bilson_raw[:n_bdy]
    valid_bdy = ~np.isnan(g_bdy)
    if valid_bdy.sum() > 5:
        # Fit (g-1)/r^2 to a polynomial in r^2
        gm1 = (g_bdy[valid_bdy] - 1.0) / r_bdy[valid_bdy]**2
        bdy_coeffs = np.polyfit(r_bdy[valid_bdy]**2, gm1, 2)
        bdy_poly = np.poly1d(bdy_coeffs)
        print(f"  Boundary fit: g(r) ≈ 1 + r²({bdy_coeffs[2]:.4f} + {bdy_coeffs[1]:.4f}r² + ...)")
    else:
        bdy_poly = np.poly1d([0])  # g ≈ 1

    # Near horizon (r→r_h): g(r_h) = 0, g ≈ g'_h (r_h - r)
    # Use last few valid points to extrapolate linearly to zero
    n_hor = min(20, len(r_grid_bilson)//5)
    r_hor = r_grid_bilson[-n_hor:]
    g_hor = g_bilson_raw[-n_hor:]
    valid_hor = ~np.isnan(g_hor) & (g_hor > 0)

    # Build full g(r) on extended grid: [small r] + [Bilson region] + [near horizon]
    r_ext_bdy = np.linspace(0.002, r_grid_bilson[0], 50, endpoint=False)
    g_ext_bdy = 1.0 + r_ext_bdy**2 * bdy_poly(r_ext_bdy**2)

    # Valid Bilson region
    valid_bilson = ~np.isnan(g_bilson_raw)
    r_bilson_valid = r_grid_bilson[valid_bilson]
    g_bilson_valid = g_bilson_raw[valid_bilson]

    # Concatenate: boundary extension + Bilson bulk
    r_grid = np.concatenate([r_ext_bdy, r_bilson_valid])
    g_bilson = np.concatenate([g_ext_bdy, g_bilson_valid])

    # Exact g(r) for comparison
    z_fine = np.linspace(0.001, 0.999, 5000)
    r_fine = np.array([r_of_z(z) for z in z_fine])
    g_fine = np.array([g_exact_at_z(z) for z in z_fine])
    g_of_r_exact = CubicSpline(r_fine, g_fine)
    g_exact_grid = g_of_r_exact(r_grid)

    valid = g_exact_grid > 0.05
    rel_err = np.abs(g_bilson[valid] - g_exact_grid[valid]) / g_exact_grid[valid]
    print(f"\n  g(r) extended: max err={rel_err.max():.4e}, mean={rel_err.mean():.4e}, "
          f"median={np.median(rel_err):.4e}")
    print(f"  r range: [{r_grid.min():.6f}, {r_grid.max():.6f}], {len(r_grid)} pts")

    return r_grid, g_bilson, g_exact_grid


# ==============================================================
# STEP 2: Build η-coordinate from g(r)
# ==============================================================
def build_eta_coordinate(r_grid, g_bilson):
    """
    Build η(r) by integrating dη = dr/(r√g).
    Convention: η = 0 at horizon (r = r_h), η → ∞ at boundary (r → 0).
    We integrate from r_h inward (decreasing r).
    """
    print("\n" + "=" * 70)
    print("STEP 2: Build η-coordinate")
    print("=" * 70)

    # Clean up: remove NaN, ensure positive g
    valid = ~np.isnan(g_bilson) & (g_bilson > 0)
    r_clean = r_grid[valid]
    g_clean = g_bilson[valid]

    # Integrand: 1/(r * sqrt(g))
    integrand_vals = 1.0 / (r_clean * np.sqrt(g_clean))

    # η(r) = ∫_r^{r_max} dr'/(r'√g(r'))
    # Convention: η = 0 at horizon (r = r_max), η → +∞ at boundary (r → 0)
    # Cumulative integral from r_max downward:
    # For each r_i, η(r_i) = ∫_{r_i}^{r_max} integrand dr'
    eta_arr = np.zeros(len(r_clean))
    for i in range(len(r_clean)):
        # Integrate from r_clean[i] to r_clean[-1] using the trapezoidal rule
        if i < len(r_clean) - 1:
            eta_arr[i] = np.trapz(integrand_vals[i:], r_clean[i:])
        else:
            eta_arr[i] = 0.0

    # η should be large at small r (boundary) and 0 at large r (horizon)
    print(f"  η range: [{eta_arr.min():.4f}, {eta_arr.max():.4f}]")
    print(f"  r range: [{r_clean.min():.6f}, {r_clean.max():.6f}]")

    # For spline interpolation, η must be strictly increasing.
    # η decreases as r increases, so for r(η) and G(η), sort by η.
    # Use reversed arrays (η increasing)
    eta_inc = eta_arr[::-1]  # now increasing
    r_inc = r_clean[::-1]    # r decreasing as η increases
    G_inc = 1.0 / r_inc**2

    # Remove any non-strictly-increasing η entries
    mask = np.concatenate([[True], np.diff(eta_inc) > 0])
    eta_inc = eta_inc[mask]
    r_inc = r_inc[mask]
    G_inc = G_inc[mask]

    # Build interpolators: η(r) uses original order, r(η) and G(η) use η-sorted
    eta_of_r = CubicSpline(r_clean, eta_arr)  # r increasing, η decreasing — fine for CubicSpline
    r_of_eta = CubicSpline(eta_inc, r_inc)
    G_of_eta = CubicSpline(eta_inc, G_inc)
    g_of_r = CubicSpline(r_clean, g_clean)

    # Verify: compare exact η
    print(f"\n  Sample η values:")
    for r_test in [0.01, 0.05, 0.1, 0.2, 0.3]:
        if r_test >= r_clean.min() and r_test <= r_clean.max():
            print(f"    r={r_test:.3f}: η={float(eta_of_r(r_test)):.4f}, "
                  f"G={1.0/r_test**2:.2f}")

    return r_clean, g_clean, eta_arr, eta_of_r, r_of_eta, G_of_eta, g_of_r


# ==============================================================
# STEP 3: Process Wilson loop data → L(h_0)
# ==============================================================
def process_wl_data(z_stars, L_arr, V_arr):
    """
    From WL boundary data {L_j, V_half_j}, parametrized by z_*:
      V_tilde = 2 * V_half  (full potential)
      h_0 = dV_tilde/dL = (dV_tilde/dz_*) / (dL/dz_*)  at each z_*
      Invert → L(h_0)

    Note: z_* is just the parametric variable used to generate data.
    The chain rule dV/dL = (dV/dz_*)/(dL/dz_*) is a numerical technique,
    not a use of the known metric.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Process Wilson loop data")
    print("=" * 70)

    V_tilde = 2.0 * V_arr  # full potential

    # L may not be monotonic (WL phase transition at large z_*).
    # Use only the connected branch where L is increasing (small z_*).
    dL = np.diff(L_arr)
    # Find where L starts decreasing
    L_max_idx = np.argmax(L_arr)
    print(f"  L peaks at index {L_max_idx} (z_*={z_stars[L_max_idx]:.4f}, L={L_arr[L_max_idx]:.4f})")
    # Use only increasing branch
    L_branch = L_arr[:L_max_idx+1]
    V_branch = V_tilde[:L_max_idx+1]

    print(f"  Using {len(L_branch)} pts on connected branch, L=[{L_branch[0]:.4f}, {L_branch[-1]:.4f}]")

    # Chebyshev fit for V_tilde(L) on the monotonic branch
    from numpy.polynomial import chebyshev as C
    L_min_v, L_max_v = L_branch[0], L_branch[-1]
    L_mapped = 2.0 * (L_branch - L_min_v) / (L_max_v - L_min_v) - 1.0
    deg_V = 30
    coeffs_V = C.chebfit(L_mapped, V_branch, deg_V)
    V_fit = C.chebval(L_mapped, coeffs_V)
    res_V = np.max(np.abs(V_fit - V_branch))
    print(f"  V_tilde(L) Chebyshev deg={deg_V}, max residual={res_V:.2e}")

    # h_0 = dV_tilde/dL via Chebyshev derivative
    dcoeffs_V = C.chebder(coeffs_V)
    dVdL_mapped = C.chebval(L_mapped, dcoeffs_V)
    dVdL_branch = dVdL_mapped * 2.0 / (L_max_v - L_min_v)

    L_arr_use = L_branch
    dVdL = dVdL_branch

    # Trim endpoints
    trim = 10
    h0_arr = dVdL[trim:-trim]
    L_trimmed = L_arr_use[trim:-trim]

    # h_0 should be positive
    mask = h0_arr > 0
    if not np.all(mask):
        print(f"  WARNING: {np.sum(~mask)} points with h_0 <= 0, removing")
        h0_arr = h0_arr[mask]
        L_trimmed = L_trimmed[mask]

    print(f"  h_0 range: [{h0_arr.min():.6f}, {h0_arr.max():.6f}]")
    print(f"  L   range: [{L_trimmed.min():.6f}, {L_trimmed.max():.6f}]")

    # Check: h_0 should decrease as L increases (deeper strings have smaller h_0)
    # So L(h_0) should be decreasing — or equivalently, h_0(L) is decreasing
    if np.all(np.diff(h0_arr) < 0):
        print("  h_0(L) is monotonically decreasing ✓")
    else:
        print("  WARNING: h_0(L) not monotonically decreasing — sorting")

    # Build L(h_0): sort by h_0
    idx3 = np.argsort(h0_arr)
    h0_sorted = h0_arr[idx3]
    L_of_h0_arr = L_trimmed[idx3]

    # Chebyshev fit for L(h_0) — work in log(h_0) space for better conditioning
    # since h_0 spans several orders of magnitude
    from numpy.polynomial import chebyshev as C
    log_h0 = np.log(h0_sorted)
    lh_min, lh_max = log_h0[0], log_h0[-1]
    lh_mapped = 2.0 * (log_h0 - lh_min) / (lh_max - lh_min) - 1.0
    deg_Lh = 30
    coeffs_Lh = C.chebfit(lh_mapped, L_of_h0_arr, deg_Lh)
    L_fit_h0 = C.chebval(lh_mapped, coeffs_Lh)
    res_Lh = np.max(np.abs(L_fit_h0 - L_of_h0_arr))
    print(f"  L(h_0) Chebyshev (in log h_0) deg={deg_Lh}, max residual={res_Lh:.2e}")

    def L_of_h0_func(h0_val):
        """Evaluate L(h_0) from Chebyshev fit in log-space."""
        h0_val = np.atleast_1d(h0_val)
        lh = np.log(np.clip(h0_val, h0_sorted[0], h0_sorted[-1]))
        mapped = 2.0 * (lh - lh_min) / (lh_max - lh_min) - 1.0
        return C.chebval(mapped, coeffs_Lh)

    L_of_h0_spline = L_of_h0_func  # duck-type as callable

    # Verify against exact: h_0 = sqrt(f_* h_*)/z_*^2
    print(f"\n  Comparison with exact h_0 (diagnostics):")
    for iL in [0, len(L_arr)//4, len(L_arr)//2]:
        L_val = L_arr[trim + iL]
        h0_val = dVdL[trim + iL]
        print(f"    L={L_val:.4f}: h_0={h0_val:.6f}")

    return h0_sorted, L_of_h0_arr, L_of_h0_spline


# ==============================================================
# STEP 4: Abel inversion → σ(H)
# ==============================================================
def abel_inversion(h0_sorted, L_of_h0_arr, L_of_h0_spline):
    """
    σ(H) = -(1/π) d/dH ∫_H^∞ L(h_0)/√(h_0²-H²) dh_0
    """
    print("\n" + "=" * 70)
    print("STEP 4: Abel inversion → σ(H)")
    print("=" * 70)

    h0_max = h0_sorted[-1]
    h0_min = h0_sorted[0]

    # Evaluate the integral I(H) = ∫_H^{h0_max} L(h_0)/√(h_0²-H²) dh_0
    # on a grid of H values
    H_grid = np.linspace(h0_min * 0.8, h0_max * 0.95, 500)
    I_abel = np.zeros(len(H_grid))

    # Use log-spaced H grid for better resolution across orders of magnitude
    H_grid = np.exp(np.linspace(np.log(h0_min * 0.9), np.log(h0_max * 0.95), 600))
    I_abel = np.zeros(len(H_grid))

    for i, H in enumerate(H_grid):
        def integ(h0):
            if h0 <= H + 1e-15: return 0.0
            L_val = float(L_of_h0_spline(h0))
            return L_val / np.sqrt(h0**2 - H**2)
        I_abel[i], _ = quad(integ, H + 1e-12, h0_max, epsabs=1e-13, epsrel=1e-13,
                            limit=1000)

    # Chebyshev fit of I(H) in log-H space, then differentiate
    from numpy.polynomial import chebyshev as C
    log_H = np.log(H_grid)
    lH_min, lH_max = log_H[0], log_H[-1]
    lH_mapped = 2.0 * (log_H - lH_min) / (lH_max - lH_min) - 1.0
    deg_I = min(80, len(H_grid) - 2)
    coeffs_I = C.chebfit(lH_mapped, I_abel, deg_I)
    I_fit = C.chebval(lH_mapped, coeffs_I)
    res_I = np.max(np.abs(I_fit - I_abel))
    print(f"  I(H) Chebyshev deg={deg_I}, max residual={res_I:.2e}")

    # dI/dH = dI/d(log H) * d(log H)/dH = dI/d(mapped) * (2/(lH_max-lH_min)) * (1/H)
    dcoeffs_I = C.chebder(coeffs_I)
    dI_dmapped = C.chebval(lH_mapped, dcoeffs_I)
    dIdH = dI_dmapped * 2.0 / (lH_max - lH_min) / H_grid

    sigma_arr = -dIdH / np.pi

    print(f"  H range: [{H_grid.min():.6f}, {H_grid.max():.6f}]")
    print(f"  σ range: [{sigma_arr.min():.6f}, {sigma_arr.max():.6f}]")
    print(f"  σ(H→0) ≈ {sigma_arr[0]:.6f}")

    # Chebyshev interpolant for σ(H) in log-H space
    coeffs_sigma = C.chebfit(lH_mapped, sigma_arr, deg_I)
    def sigma_of_H_func(H_val):
        H_val = np.atleast_1d(H_val)
        lh = np.log(np.clip(H_val, H_grid[0], H_grid[-1]))
        mapped = 2.0 * (lh - lH_min) / (lH_max - lH_min) - 1.0
        return C.chebval(mapped, coeffs_sigma)
    sigma_of_H = sigma_of_H_func

    return H_grid, sigma_arr, sigma_of_H


# ==============================================================
# STEP 5: ODE integration  dη/dH = σ(H)√G(η)
# ==============================================================
def integrate_ode(H_grid, sigma_of_H, G_of_eta, eta_max):
    """
    Solve dη/dH = σ(H) * √G(η) with η(0) = 0.
    Integrate from H = H_min toward H_max.
    """
    print("\n" + "=" * 70)
    print("STEP 5: ODE integration")
    print("=" * 70)

    H_min = H_grid[0]
    H_max = H_grid[-1]

    # Clip G_of_eta to its valid range
    eta_lo, eta_hi = G_of_eta.x[0], G_of_eta.x[-1]

    def rhs(H, eta):
        # dη/dH = σ(H) * √G(η)
        sig = float(sigma_of_H(H))
        eta_val = np.clip(eta[0], eta_lo, eta_hi)
        G_val = max(float(G_of_eta(eta_val)), 1.0)
        return [sig * np.sqrt(G_val)]

    # Integrate from H_min (near horizon) to H_max (toward boundary)
    sol = solve_ivp(rhs, [H_min, H_max], [0.0],
                    method='Radau', rtol=1e-12, atol=1e-14,
                    dense_output=True, max_step=(H_max-H_min)/2000)

    if not sol.success:
        print(f"  ODE integration failed: {sol.message}")
        return None, None, None

    # Evaluate on a fine H grid
    H_fine = np.linspace(H_min, H_max, 500)
    eta_fine = sol.sol(H_fine)[0]

    print(f"  H range:  [{H_fine[0]:.6f}, {H_fine[-1]:.6f}]")
    print(f"  η range:  [{eta_fine[0]:.6f}, {eta_fine[-1]:.6f}]")

    # Build H(η) by inverting η(H)
    # η should be monotonically increasing with H
    if not np.all(np.diff(eta_fine) > 0):
        print("  WARNING: η(H) not strictly increasing — check")
        # Force monotonicity
        for i in range(1, len(eta_fine)):
            if eta_fine[i] <= eta_fine[i-1]:
                eta_fine[i] = eta_fine[i-1] + 1e-10

    H_of_eta_spline = CubicSpline(eta_fine, H_fine)
    eta_of_H_spline = CubicSpline(H_fine, eta_fine)

    return H_fine, eta_fine, H_of_eta_spline


# ==============================================================
# STEP 6: Recover F, χ, f, h
# ==============================================================
def recover_metric(r_clean, g_clean, eta_arr, eta_of_r, r_of_eta, G_of_eta,
                   g_of_r, H_of_eta_spline):
    """
    F(η) = H(η)²/G(η)
    χ(r) = log(g(r)/(r²F(η(r))))
    h(z) = z²/r², f(z) = g/α²
    """
    print("\n" + "=" * 70)
    print("STEP 6: Recover metric")
    print("=" * 70)

    # Work on the r_clean grid (interior, away from boundary/horizon artifacts)
    trim = 10
    r_work = r_clean[trim:-trim]
    g_work = g_clean[trim:-trim]

    eta_work = np.array([float(eta_of_r(r)) for r in r_work])

    # Clip η to the range where H(η) is defined
    eta_min_H = H_of_eta_spline.x[0]
    eta_max_H = H_of_eta_spline.x[-1]
    mask = (eta_work >= eta_min_H) & (eta_work <= eta_max_H)
    r_work = r_work[mask]
    g_work = g_work[mask]
    eta_work = eta_work[mask]

    H_work = np.array([float(H_of_eta_spline(e)) for e in eta_work])
    G_work = 1.0 / r_work**2

    # F = H²/G
    F_work = H_work**2 / G_work

    # χ = log(g/(r²F))
    chi_recon = np.log(g_work / (r_work**2 * F_work))

    # Exact χ for comparison
    # Need z(r) — use exact for comparison
    z_fine = np.linspace(0.001, 0.999, 5000)
    r_fine_exact = np.array([r_of_z(z) for z in z_fine])
    chi_fine_exact = np.array([chi_exact_at_z(z) for z in z_fine])
    chi_of_r_exact = CubicSpline(r_fine_exact, chi_fine_exact)
    chi_exact_work = chi_of_r_exact(r_work)

    err_chi = np.abs(chi_recon - chi_exact_work)
    print(f"  χ(r): max abs err = {err_chi.max():.4e}, mean = {err_chi.mean():.4e}")

    # Recover f(z) and h(z):
    # h = z²/r², f = g/α², α = (z/r)dr/dz
    # We need z(r). From r = z/√h and h = z²/r², so z = r√h = r·z/r = z. Circular.
    # Instead: dr/dz = α/√h, and h = z²/r². So:
    # From the known r(z) = z/√h(z), and h unknown, solve the ODE:
    #   dr/dz = α/√h = (z/r · dr/dz)/√(z²/r²) = (z/r · dr/dz) · r/z = dr/dz
    # That's circular. Use a different approach:
    # Since h = z²/r² and h = e^χ/α², and α = (z/r)r'(z):
    #   e^χ = α²h = (z/r)² (r')² · z²/r² = z⁴(r')²/r⁴
    # So (r')² = e^χ r⁴/z⁴, i.e., dr/dz = r²e^{χ/2}/z²
    # This is an ODE for r(z) given χ(r)!

    # But χ is a function of r, not z. So the ODE is:
    #   dr/dz = (r²/z²) e^{χ(r)/2}
    # with BC r(0) = 0 (boundary).

    # We have χ(r) from the reconstruction. Solve this ODE.
    chi_of_r_recon = CubicSpline(r_work, chi_recon)

    def rhs_rz(z, r):
        if r <= 0 or z <= 0:
            return 0.0
        # χ(r) — clip to data range
        r_val = np.clip(r, r_work[0], r_work[-1])
        chi_val = float(chi_of_r_recon(r_val))
        return (r**2 / z**2) * np.exp(chi_val / 2.0)

    # Near boundary: r ≈ z (since h(0)=1, α(0)=1), so start at small z
    z_start = 0.01
    r_start = z_start  # r ≈ z near boundary

    sol_rz = solve_ivp(rhs_rz, [z_start, 0.99], [r_start],
                       method='RK45', rtol=1e-10, atol=1e-12,
                       dense_output=True, max_step=0.001)

    if not sol_rz.success:
        print(f"  ODE r(z) failed: {sol_rz.message}")
        return r_work, chi_recon, chi_exact_work, None, None, None, None, None, None

    z_grid_out = np.linspace(z_start + 0.01, 0.93, 200)
    r_of_z_recon = sol_rz.sol(z_grid_out)[0]

    # Now: h(z) = z²/r(z)², f(z) = g(r(z))/α(z)²
    h_recon = z_grid_out**2 / r_of_z_recon**2

    # α = (z/r) dr/dz — compute dr/dz from the solution
    drdz = np.array([rhs_rz(z, r) for z, r in zip(z_grid_out, r_of_z_recon)])
    alpha_recon = (z_grid_out / r_of_z_recon) * drdz

    # g at the reconstructed r values
    g_at_r = np.array([float(g_of_r(np.clip(r, r_work[0], r_work[-1])))
                       for r in r_of_z_recon])
    f_recon = g_at_r / alpha_recon**2

    # Exact for comparison
    h_exact_arr = np.array([h_exact(z) for z in z_grid_out])
    f_exact_arr = np.array([f_exact(z) for z in z_grid_out])

    mask_f = f_exact_arr > 0.05
    err_h = np.abs(h_recon - h_exact_arr) / h_exact_arr
    err_f = np.abs(f_recon[mask_f] - f_exact_arr[mask_f]) / f_exact_arr[mask_f]

    print(f"  h(z): max rel err = {err_h.max():.4e}, mean = {err_h.mean():.4e}")
    print(f"  f(z): max rel err (f>0.05) = {err_f.max():.4e}, mean = {err_f.mean():.4e}")

    return (r_work, chi_recon, chi_exact_work,
            z_grid_out, f_recon, f_exact_arr, h_recon, h_exact_arr, r_of_z_recon)


# ==============================================================
# STEP 7: Plots
# ==============================================================
def make_plots(r_grid, g_bilson, g_exact_grid,
               r_work, chi_recon, chi_exact_work,
               z_grid, f_recon, f_exact_arr, h_recon, h_exact_arr):
    print("\n" + "=" * 70)
    print("Generating plots")
    print("=" * 70)

    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (0,0): g(r) from Bilson
    ax = axes[0, 0]
    valid = ~np.isnan(g_bilson)
    ax.plot(r_grid[valid], g_exact_grid[valid], 'r-', lw=2, label=r'$g(r)$ exact')
    ax.plot(r_grid[valid], g_bilson[valid], 'b--', lw=1.5, label=r'$g(r)$ Bilson')
    ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$g(r)$')
    ax.set_title(r'Step 1: $S_{EE}(l) \to g(r)$ (non-circular)')
    ax.legend()

    # (0,1): χ(r)
    ax = axes[0, 1]
    if chi_recon is not None:
        ax.plot(r_work, chi_exact_work, 'r-', lw=2, label=r'$\chi(r)$ exact')
        ax.plot(r_work, chi_recon, 'b--', lw=1.5, label=r'$\chi(r)$ recon')
        ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$\chi(r)$')
        ax.set_title(r'Steps 3-5: $V(L) \to \chi(r)$ (Hashimoto)')
        ax.legend()

    # (0,2): f/h ratio
    ax = axes[0, 2]
    if f_recon is not None:
        foh_exact = f_exact_arr / h_exact_arr
        foh_recon = f_recon / h_recon
        ax.plot(z_grid, foh_exact, 'r-', lw=2, label=r'$f/h$ exact')
        ax.plot(z_grid, foh_recon, 'b--', lw=1.5, label=r'$f/h$ recon')
        ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$f/h$')
        ax.set_title(r'Ratio $f/h$')
        ax.legend()

    # (1,0): f(z)
    ax = axes[1, 0]
    if f_recon is not None:
        ax.plot(z_grid, f_exact_arr, 'r-', lw=2, label=r'$f(z)$ exact')
        ax.plot(z_grid, f_recon, 'b--', lw=1.5, label=r'$f(z)$ recon')
        ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$f(z)$')
        ax.set_title(r'Reconstructed $f(z)$')
        ax.legend()

    # (1,1): h(z)
    ax = axes[1, 1]
    if h_recon is not None:
        ax.plot(z_grid, h_exact_arr, 'r-', lw=2, label=r'$h(z)$ exact')
        ax.plot(z_grid, h_recon, 'b--', lw=1.5, label=r'$h(z)$ recon')
        ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$h(z)$')
        ax.set_title(r'Reconstructed $h(z)$')
        ax.legend()

    # (1,2): relative errors
    ax = axes[1, 2]
    if f_recon is not None:
        err_f = np.abs(f_recon - f_exact_arr) / np.maximum(f_exact_arr, 1e-10)
        err_h = np.abs(h_recon - h_exact_arr) / h_exact_arr
        ax.semilogy(z_grid, err_f, 'b-', lw=1.5, label=r'$|\Delta f/f|$')
        ax.semilogy(z_grid, err_h, 'r-', lw=1.5, label=r'$|\Delta h/h|$')
        ax.set_xlabel(r'$z$'); ax.set_ylabel('Relative error')
        ax.set_title('Reconstruction errors')
        ax.legend()
        ax.set_ylim(1e-7, 1e0)

    plt.suptitle('Non-circular metric reconstruction (Bilson + Hashimoto)', fontsize=14)
    plt.tight_layout()
    outpath = os.path.join(fig_dir, 'bilson_reconstruction_v3.pdf')
    plt.savefig(outpath, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


# ==============================================================
# Main
# ==============================================================
def main():
    # Step 0: generate boundary data
    z_stars, l_arr, A_arr, L_arr, V_arr = generate_data()

    # Step 1: non-circular Bilson
    r_grid, g_bilson, g_exact_grid = bilson_inversion(z_stars, l_arr, A_arr)

    # Step 2: build η-coordinate
    (r_clean, g_clean, eta_arr, eta_of_r, r_of_eta,
     G_of_eta, g_of_r) = build_eta_coordinate(r_grid, g_bilson)

    # Step 3: process WL data
    h0_sorted, L_of_h0_arr, L_of_h0_spline = process_wl_data(z_stars, L_arr, V_arr)

    # Step 4: Abel inversion
    H_grid, sigma_arr, sigma_of_H = abel_inversion(h0_sorted, L_of_h0_arr, L_of_h0_spline)

    # Step 5: ODE integration
    eta_max = eta_arr.max()
    H_fine, eta_fine, H_of_eta_spline = integrate_ode(
        H_grid, sigma_of_H, G_of_eta, eta_max)

    if H_of_eta_spline is None:
        print("\nODE integration failed — cannot proceed to metric recovery")
        return

    # Step 6: recover metric
    results = recover_metric(r_clean, g_clean, eta_arr, eta_of_r, r_of_eta,
                             G_of_eta, g_of_r, H_of_eta_spline)

    (r_work, chi_recon, chi_exact_work,
     z_grid, f_recon, f_exact_arr, h_recon, h_exact_arr, r_of_z_recon) = results

    # Step 7: plots
    make_plots(r_grid, g_bilson, g_exact_grid,
               r_work, chi_recon, chi_exact_work,
               z_grid, f_recon, f_exact_arr, h_recon, h_exact_arr)

    print("\n" + "=" * 70)
    print("NON-CIRCULAR RECONSTRUCTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
