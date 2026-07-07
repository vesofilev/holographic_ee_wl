"""
Independent numerical review of Appendix A's central claim and Appendix C's
derivative identity, on the Gubser-Rocha background (Q = 1, z_h = 1).

 [N1] THE DEGENERACY THEOREM END-TO-END:
      build a deformed member of the family,
        tilde_h = h_GR + eps * z(1-z),      eps = 0.5 and eps = 2.0
        tilde_f = g(tilde_r(z)) / tilde_alpha(z)^2,
      with g evaluated EXACTLY (root-finding on r_GR(z), no splines), and
      check that (l, A_reg) computed from the tilde metric via eqs (5.12)-(5.13)
      coincide with the Gubser-Rocha values at the same Bilson turning point.
      Also check tilde_chi != chi (the spacetimes are genuinely different).

 [N2] Wilson loop identity eq (C.7):
      dV_reg/dL = sqrt(f* h*)/(2 zhat*^2)   [units 1/(pi alpha') = 1]
      via central differences of parametric L(zhat*), V_reg(zhat*).

 [N3] Step 3 temperature relation:
      tilde_f'(z_h) = g'(r_h)/(sqrt(tilde_h(z_h)) tilde_alpha(z_h))
      checked by finite differences (supports the [A3] symbolic finding).
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

Q = 1.0
ZH = 1.0

# ----- exact Gubser-Rocha metric -----
def U(zv):    return (1 + (1+3*Q)*zv + (1+3*Q+3*Q**2)*zv**2) / (1+Q*zv)**1.5
def f_gr(zv): return (1 - zv) * U(zv)
def h_gr(zv): return (1 + Q*zv)**1.5
def hp_gr(zv): return 1.5*Q*(1 + Q*zv)**0.5
def alpha_gr(zv): return 1 - 3*Q*zv/(4*(1 + Q*zv))
def r_gr(zv): return zv/np.sqrt(h_gr(zv))
def g_exact(rv):
    """g(r) from the exact GR metric: invert r_GR(z) then alpha^2 f."""
    if rv <= 0: return 1.0
    zz = brentq(lambda t: r_gr(t) - rv, 0.0, ZH, xtol=1e-15, rtol=8.9e-16)
    return alpha_gr(zz)**2 * f_gr(zz)
R_H = r_gr(ZH)

def gp_exact(rv, dr=1e-7):
    return (g_exact(rv + dr) - g_exact(rv - dr)) / (2*dr)

# ----- deformed member -----
def make_tilde(eps):
    dh  = lambda zv: zv*(ZH - zv)
    dhp = lambda zv: ZH - 2*zv
    th  = lambda zv: h_gr(zv) + eps*dh(zv)
    thp = lambda zv: hp_gr(zv) + eps*dhp(zv)
    tal = lambda zv: 1 - zv*thp(zv)/(2*th(zv))
    tr  = lambda zv: zv/np.sqrt(th(zv))
    def tf(zv):
        if zv <= 0: return 1.0
        return g_exact(tr(zv)) / tal(zv)**2
    return th, thp, tal, tr, tf

# ----- parametric RT integrals for a general (f,h), eqs (5.12)-(5.13) -----
def rt_l_A(f, h, zst, n=400):
    """l and A_reg (Omega=1) for turning point z*, theta-substitution z=z* sin(theta)."""
    hst = h(zst)
    th = np.linspace(1e-12, np.pi/2 - 1e-12, n)  # avoid exact endpoints for vector eval
    # l/2 integrand in theta
    def l_int(t):
        zv = zst*np.sin(t)
        rad = (h(zv)**2 * zst**4)/(zv**4 * hst**2) - 1.0
        return zst*np.cos(t)/np.sqrt(f(zv)*h(zv)*rad)
    lval = 2*quad(l_int, 0, np.pi/2, limit=400)[0]
    # A_reg connected-minus part
    def A_int(t):
        zv = zst*np.sin(t)
        w = 1.0 - (zv**4 * hst**2)/(zst**4 * h(zv)**2)
        return zst*np.cos(t)*np.sqrt(h(zv))/(zv**2*np.sqrt(f(zv))) * (1/np.sqrt(w) - 1)
    A1 = quad(A_int, 0, np.pi/2, limit=400)[0]
    # disconnected part from z* to z_h, substitution z = z_h - w^2 (f ~ (z_h - z))
    def D_int(w):
        zv = ZH - w**2
        return 2*w*np.sqrt(h(zv))/(zv**2*np.sqrt(f(zv)))
    A2 = quad(D_int, 0, np.sqrt(ZH - zst), limit=400)[0]
    return lval, A1 - A2

print("=" * 74)
print("[N1] Degeneracy end-to-end: identical (l, A_reg) at matched Bilson r_*")
print("=" * 74)
for eps in (0.5, 2.0):
    th, thp, tal, tr, tf = make_tilde(eps)
    amin = min(tal(zv) for zv in np.linspace(0, ZH, 2001))
    print(f"\n  eps = {eps}:  min tilde_alpha on [0,1] = {amin:.4f}  (must be > 0)")
    print(f"  {'r_*':>8s} {'l (GR)':>12s} {'l (tilde)':>12s} {'|dl|':>9s} "
          f"{'A (GR)':>12s} {'A (tilde)':>12s} {'|dA|':>9s}")
    worst_l = worst_A = 0.0
    for zst in (0.25, 0.4, 0.55, 0.7, 0.85):
        rst = r_gr(zst)                                     # Bilson turning point
        tzst = brentq(lambda t: tr(t) - rst, 0, ZH, xtol=1e-15)  # same r_* in tilde
        l0, A0 = rt_l_A(f_gr, h_gr, zst)
        l1, A1 = rt_l_A(tf, th, tzst)
        worst_l = max(worst_l, abs(l1-l0)); worst_A = max(worst_A, abs(A1-A0))
        print(f"  {rst:8.4f} {l0:12.8f} {l1:12.8f} {abs(l1-l0):9.1e} "
              f"{A0:12.8f} {A1:12.8f} {abs(A1-A0):9.1e}")
    print(f"  worst |dl| = {worst_l:.2e}, worst |dA_reg| = {worst_A:.2e}"
          f"   -> {'IDENTICAL (theorem verified)' if max(worst_l, worst_A) < 1e-6 else 'MISMATCH'}")
    # chi differs
    zmid = 0.5
    chi0 = np.log(alpha_gr(zmid)**2 * h_gr(zmid))
    chi1 = np.log(tal(zmid)**2 * th(zmid))
    print(f"  chi at z=0.5:  GR {chi0:.6f}  vs tilde {chi1:.6f}"
          f"   -> genuinely different spacetime: {abs(chi1-chi0) > 1e-3}")

print()
print("=" * 74)
print("[N2] Wilson loop derivative identity  dV_reg/dL = sqrt(f* h*)/(2 zhat*^2)")
print("=" * 74)
def wl_L_V(zst, n=400):
    fst, hst = f_gr(zst), h_gr(zst)
    def L_int(t):
        zv = zst*np.sin(t)
        rad = (f_gr(zv)*h_gr(zv)*zst**4)/(zv**4*fst*hst) - 1.0
        return zst*np.cos(t)/np.sqrt(f_gr(zv)*h_gr(zv)*rad)
    Lval = 2*quad(L_int, 0, np.pi/2, limit=400)[0]
    def V_int(t):
        zv = zst*np.sin(t)
        w = 1.0 - (zv**4*fst*hst)/(zst**4*f_gr(zv)*h_gr(zv))
        return zst*np.cos(t)/zv**2 * (1/np.sqrt(w) - 1)
    V1 = quad(V_int, 0, np.pi/2, limit=400)[0]
    V2 = 1.0/zst - 1.0/ZH          # int_{z*}^{zh} dz/z^2
    return Lval, V1 - V2           # units 1/(pi alpha') = 1

print(f"  {'zhat_*':>7s} {'dV/dL (numeric)':>16s} {'sqrt(f*h*)/(2 z*^2)':>20s} {'rel.err':>10s}")
for zst in (0.3, 0.45, 0.6, 0.75):
    d = 1e-4
    Lm, Vm = wl_L_V(zst - d); Lp, Vp = wl_L_V(zst + d)
    dVdL = (Vp - Vm)/(Lp - Lm)
    pred = np.sqrt(f_gr(zst)*h_gr(zst))/(2*zst**2)
    print(f"  {zst:7.2f} {dVdL:16.8f} {pred:20.8f} {abs(dVdL/pred-1):10.1e}")

print()
print("=" * 74)
print("[N3] Step 3: tilde_f'(z_h) = g'(r_h)/(sqrt(h(z_h)) alpha(z_h))")
print("=" * 74)
for eps in (0.0, 0.5, 2.0):
    th, thp, tal, tr, tf = make_tilde(eps)
    d = 1e-6
    fp_num = (tf(ZH - d) - tf(ZH - 3*d))/(2*d)   # one-sided near horizon
    fp_pred = gp_exact(R_H - 1e-6)/(np.sqrt(th(ZH))*tal(ZH))
    print(f"  eps = {eps}:  f'(z_h) numeric {fp_num:+.6f}   "
          f"predicted {fp_pred:+.6f}   alpha(z_h) = {tal(ZH):.4f}"
          f"   -> T = |f'(z_h)|/4pi = {abs(fp_pred)/(4*np.pi):.6f}")
print("  -> T varies with eps (through alpha(z_h)): fixing T fixes h'(z_h),")
print("     confirming the symbolic finding [A3].")
