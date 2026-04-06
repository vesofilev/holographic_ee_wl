"""
Generate high-precision boundary data {l, S_EE, L, V} for the Bilson reconstruction.

All turning-point integrals use the substitution t = sin²θ to remove
endpoint singularities, giving smooth integrands on [0, π/2].
This allows quad to reach machine precision (~1e-15).
"""
import numpy as np
from scipy.integrate import quad

Q = 1.0
Z_H = 1.0

def f_exact(z):
    U = (1 + (1+3*Q)*z + (1+3*Q+3*Q**2)*z**2) / (1+Q*z)**1.5
    return (1 - z) * U

def h_exact(z):
    return (1 + Q*z) ** 1.5

def r_of_z(z):
    return z / np.sqrt(h_exact(z))

def g_exact_at_z(z):
    alpha = 1.0 - z * 1.5*Q*(1+Q*z)**0.5 / (2*(1+Q*z)**1.5)
    return alpha**2 * f_exact(z)

def chi_exact_at_z(z):
    alpha = 1.0 - z * 1.5*Q*(1+Q*z)**0.5 / (2*(1+Q*z)**1.5)
    return np.log(alpha**2 * h_exact(z))


# === RT integrals with θ-substitution ===

def compute_l_RT(z_star):
    """l(z_*) via θ-substitution: t = sin²θ removes both endpoint singularities."""
    h_star = h_exact(z_star)
    def integrand(theta):
        st, ct = np.sin(theta), np.cos(theta)
        if st < 1e-30 or ct < 1e-30: return 0.0
        z = z_star * st
        fv, hv = f_exact(z), h_exact(z)
        t2 = st**4  # t² = sin⁴θ
        denom = hv**2 / (t2 * h_star**2) - 1.0
        if denom <= 0: return 0.0
        return z_star * ct / (np.sqrt(fv * hv) * np.sqrt(denom))
    result, _ = quad(integrand, 0, np.pi/2, epsabs=1e-15, epsrel=1e-15, limit=2000)
    return 2.0 * result

def compute_Areg_RT(z_star):
    """A_reg(z_*) = I1 - I2, both with singularity-free integrands."""
    h_star = h_exact(z_star)

    # I1: connected-minus-disconnected from 0 to z_*, θ-substitution
    def integrand1(theta):
        st, ct = np.sin(theta), np.cos(theta)
        if st < 1e-30 or ct < 1e-30: return 0.0
        z = z_star * st
        fv, hv = f_exact(z), h_exact(z)
        eta = st**4 * h_star**2 / hv**2
        if eta >= 1.0 - 1e-30: return 0.0
        prefactor = np.sqrt(hv) / (z**2 * np.sqrt(fv))
        bracket = 1.0/np.sqrt(1.0 - eta) - 1.0
        return prefactor * bracket * z_star * ct
    I1, _ = quad(integrand1, 0, np.pi/2, epsabs=1e-15, epsrel=1e-15, limit=2000)

    # I2: disconnected from z_* to z_h
    # Singularity at z_h: f ~ c(1-z), so 1/√f ~ 1/√(1-z)
    # Substitution: z = z_* + (z_h - z_*) sin²φ
    dz = Z_H - z_star
    def integrand2(phi):
        sp, cp = np.sin(phi), np.cos(phi)
        if cp < 1e-30: return 0.0
        z = z_star + dz * sp**2
        if z >= Z_H - 1e-15: return 0.0
        fv, hv = f_exact(z), h_exact(z)
        if fv <= 0: return 0.0
        # dz = 2*dz*sin(φ)*cos(φ) dφ
        return np.sqrt(hv) / (z**2 * np.sqrt(fv)) * 2 * dz * sp * cp
    I2, _ = quad(integrand2, 0, np.pi/2, epsabs=1e-15, epsrel=1e-15, limit=2000)

    return I1 - I2


# === WL integrals with θ-substitution ===

def compute_L_WL(z_star):
    f_star, h_star = f_exact(z_star), h_exact(z_star)
    F_star = f_star * h_star
    def integrand(theta):
        st, ct = np.sin(theta), np.cos(theta)
        if st < 1e-30 or ct < 1e-30: return 0.0
        z = z_star * st
        fv, hv = f_exact(z), h_exact(z)
        Fv = fv * hv
        if Fv <= 0: return 0.0
        denom = Fv / (st**4 * F_star) - 1.0
        if denom <= 0: return 0.0
        return z_star * ct / (np.sqrt(Fv) * np.sqrt(denom))
    result, _ = quad(integrand, 0, np.pi/2, epsabs=1e-15, epsrel=1e-15, limit=2000)
    return 2.0 * result

def compute_Vreg_WL(z_star):
    f_star, h_star = f_exact(z_star), h_exact(z_star)
    F_star = f_star * h_star

    # I1: connected-minus-disconnected, θ-substitution
    def integrand1(theta):
        st, ct = np.sin(theta), np.cos(theta)
        if st < 1e-30 or ct < 1e-30: return 0.0
        z = z_star * st
        fv, hv = f_exact(z), h_exact(z)
        Fv = fv * hv
        if Fv <= 0: return 0.0
        eta = st**4 * F_star / Fv
        if eta >= 1.0 - 1e-30: return 0.0
        bracket = 1.0/np.sqrt(1.0 - eta) - 1.0
        return (1.0/z**2) * bracket * z_star * ct
    I1, _ = quad(integrand1, 0, np.pi/2, epsabs=1e-15, epsrel=1e-15, limit=2000)

    I2 = 1.0/z_star - 1.0/Z_H
    return I1 - I2


# === Generate and save ===

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    z1 = np.linspace(0.005, 0.3, 300)
    z2 = np.linspace(0.3, 0.7, 200)
    z3 = np.linspace(0.7, 0.93, 100)
    z_stars = np.unique(np.concatenate([z1, z2, z3]))

    N = len(z_stars)
    l_arr = np.zeros(N)
    A_arr = np.zeros(N)
    L_arr = np.zeros(N)
    V_arr = np.zeros(N)
    r_arr = np.zeros(N)

    for i, zs in enumerate(z_stars):
        l_arr[i] = compute_l_RT(zs)
        A_arr[i] = compute_Areg_RT(zs)
        L_arr[i] = compute_L_WL(zs)
        V_arr[i] = compute_Vreg_WL(zs)
        r_arr[i] = r_of_z(zs)
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{N}")

    np.savez('bilson_data_hp.npz',
             z_stars=z_stars, l=l_arr, A=A_arr, L=L_arr, V=V_arr, r_exact=r_arr)

    # Quick precision check: compare l at a known point
    # Pure AdS limit (small z_*): l = 2√π Γ(3/4)/Γ(1/4) * z_*
    from scipy.special import gamma
    c_ads = 2*np.sqrt(np.pi)*gamma(0.75)/gamma(0.25)
    z_test = z_stars[0]
    l_ads = c_ads * z_test  # pure AdS (approximate for small z_*)
    print(f"\nPrecision check at z_*={z_test:.4f}:")
    print(f"  l computed = {l_arr[0]:.15e}")
    print(f"  l pure AdS = {l_ads:.15e}")
    print(f"  (difference expected due to finite-T corrections)")
    print(f"\nSaved {N} pts to bilson_data_hp.npz")
    print(f"  l:  [{l_arr.min():.10f}, {l_arr.max():.10f}]")
    print(f"  S:  [{(2*A_arr).min():.10f}, {(2*A_arr).max():.10f}]")
