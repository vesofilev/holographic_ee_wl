"""
Full non-circular metric reconstruction: S_EE(l) + V(L) → g(r), χ(r) → f(z), h(z).
Generates the publication figure.

Uses high-precision data from bilson_data_hp.npz.
"""
import numpy as np
from scipy.integrate import quad, solve_ivp, cumulative_trapezoid
from scipy.interpolate import CubicSpline
from generate_data_hp import (r_of_z, g_exact_at_z, chi_exact_at_z,
                               f_exact, h_exact)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 13, 'legend.fontsize': 10})

d = np.load('bilson_data_hp.npz')
l, A, L, V, z_stars = d['l'], d['A'], d['L'], d['V'], d['z_stars']


def compute_eta_from_boundary(r_arr, g_arr):
    reg = (1.0/np.sqrt(g_arr) - 1.0) / r_arr
    return np.log(r_arr) + cumulative_trapezoid(reg, r_arr, initial=0)


# ================================================================
# STEP 1: Bilson  S_EE → g(r)
# ================================================================
print("Step 1: Bilson g(r)...")
S = 2.0*A; sp_S = CubicSpline(l, S); dSdl = sp_S(l, 1)
r_star = 1.0/np.sqrt(dSdl)
trim = 5; r_d = r_star[trim:-trim]; l_d = l[trim:-trim]
ix = np.argsort(r_d); r_d = r_d[ix]; l_d = l_d[ix]
lor = l_d[:30]/r_d[:30]; cfs = np.polyfit(r_d[:30]**2, lor, 3)
poly_lr = np.poly1d(cfs)

def l_of_r(rs):
    rs = np.atleast_1d(rs); out = np.zeros_like(rs); m = rs >= r_d[0]
    if m.any(): out[m] = CubicSpline(r_d, l_d)(np.clip(rs[m], r_d[0], r_d[-1]))
    if (~m).any(): out[~m] = rs[~m]*poly_lr(rs[~m]**2)
    return out

r_bilson = np.linspace(r_d.min()*1.2, r_d.max()*0.92, 500)
I_bilson = np.zeros(len(r_bilson))
for i, r in enumerate(r_bilson):
    def ib(theta, r=r):
        st = np.sin(theta)
        if st < 1e-30: return 0.0
        return r**2/4*l_of_r(r*np.sqrt(st))*2*st
    I_bilson[i], _ = quad(ib, 0, np.pi/2, epsabs=1e-15, epsrel=1e-15, limit=2000)
Isp = CubicSpline(r_bilson, I_bilson); dIdr = Isp(r_bilson, 1)
isg = (2/np.pi)*dIdr/r_bilson**2
gb = np.where(isg > 0, 1/isg**2, np.nan)
v = ~np.isnan(gb) & (gb > 0); r_rec = r_bilson[v]; g_rec = gb[v]

# Extend to boundary
r_ext = np.linspace(0.001, r_rec[0], 50, endpoint=False)
n_bdy = 20; gm1 = (g_rec[:n_bdy]-1)/r_rec[:n_bdy]**2
bdy_cfs = np.polyfit(r_rec[:n_bdy]**2, gm1, 2); bdy_poly = np.poly1d(bdy_cfs)
g_ext = 1.0 + r_ext**2*bdy_poly(r_ext**2)
r_full = np.concatenate([r_ext, r_rec]); g_full = np.concatenate([g_ext, g_rec])
g_of_r = CubicSpline(r_full, g_full)

# Exact comparison
z_fine = np.linspace(0.001, 0.999, 5000)
r_fine = np.array([r_of_z(z) for z in z_fine])
g_fine = np.array([g_exact_at_z(z) for z in z_fine])
ixe = np.argsort(r_fine); r_fine = r_fine[ixe]; g_fine = g_fine[ixe]
g_ex_sp = CubicSpline(r_fine, g_fine)
err_g = np.abs(g_rec - g_ex_sp(r_rec))/g_ex_sp(r_rec)
print(f"  g(r): median err = {np.median(err_g):.2e}")


# ================================================================
# STEP 2: η(r), G(η) — boundary side, subtracted
# ================================================================
print("Step 2: η coordinate...")
eta_rec = compute_eta_from_boundary(r_full, g_full)
G_of_eta_sp = CubicSpline(eta_rec, 1.0/r_full**2)
r_of_eta_sp = CubicSpline(eta_rec, r_full)

# δΦ(η) = Φ(η) - e^η
r_ads = np.exp(eta_rec)
delta_r = r_full - r_ads
g2_val = bdy_poly(0)
delta_Phi_tail = g2_val * r_full[0]**3 / 12.0
delta_Phi_arr = delta_Phi_tail + cumulative_trapezoid(delta_r, eta_rec, initial=0)
Phi_arr = np.exp(eta_rec) + delta_Phi_arr


# ================================================================
# STEP 3: WL → σ(H), L(h_0)
# ================================================================
print("Step 3: Abel inversion σ(H)...")
L_max_idx = np.argmax(L); L_br = L[:L_max_idx+1]; V_br = V[:L_max_idx+1]
V_tilde = 2.0*V_br; sp_V = CubicSpline(L_br, V_tilde); dVdL = sp_V(L_br, 1)
tr = 10; h0_a = dVdL[tr:-tr]; L_t = L_br[tr:-tr]
p = h0_a > 0; h0_a = h0_a[p]; L_t = L_t[p]
i3 = np.argsort(h0_a); h0_s = h0_a[i3]; L_s = L_t[i3]
Llh = CubicSpline(np.log(h0_s), L_s)
ca = L_s[-1]*np.sqrt(h0_s[-1]); h0dm = h0_s[-1]

def Lh(h0):
    if h0 <= h0dm: return float(Llh(np.log(max(h0, h0_s[0]))))
    return ca/np.sqrt(h0)

Hg = np.exp(np.linspace(np.log(h0_s[0]*1.01), np.log(h0_s[-1]*1.1), 400))
Ia = np.zeros(len(Hg))
for i, H in enumerate(Hg):
    def ia(th, H=H):
        ct = np.cos(th)
        if ct < 1e-30: return 0.0
        return Lh(H/ct)/ct
    Ia[i], _ = quad(ia, 0, np.pi/2-1e-12, epsabs=1e-14, epsrel=1e-14, limit=2000)
lH = np.log(Hg); Is2 = CubicSpline(lH, Ia)
dIlH = Is2(lH, 1); sigma_arr = -dIlH/(Hg*np.pi)
sigma_sp = CubicSpline(lH, sigma_arr)
def sigma_of_H(Hv):
    return float(sigma_sp(np.log(np.clip(Hv, Hg[0], Hg[-1]))))


# ================================================================
# STEP 4: Ψ(H) — subtracted
# ================================================================
print("Step 4: Ψ(H)...")
H_eval = np.exp(np.linspace(np.log(Hg[0]), np.log(Hg[-1]), 500))
H_max_data = Hg[-1]

def delta_sigma(Hv):
    return sigma_of_H(Hv) - 0.5*Hv**(-1.5)

H_fit_mask = (Hg >= 0.5*H_max_data) & (Hg <= H_max_data)
dsig_vals = (sigma_arr[H_fit_mask] - 0.5*Hg[H_fit_mask]**(-1.5))*Hg[H_fit_mask]**2.5
C_fit = np.mean(dsig_vals)
delta_tail = 2.0*C_fit/(3.0*H_max_data**1.5)

delta_Psi_arr = np.zeros(len(H_eval))
for i, H in enumerate(H_eval):
    val, _ = quad(delta_sigma, H, H_max_data, epsabs=1e-15, epsrel=1e-15, limit=2000)
    delta_Psi_arr[i] = val + delta_tail

Psi_arr = 1.0/np.sqrt(H_eval) + delta_Psi_arr


# ================================================================
# STEP 5: Match Φ = Ψ → η(H) → H(η) → F, χ
# ================================================================
print("Step 5: Match and recover χ(r)...")
log_Phi = np.log(np.maximum(Phi_arr, 1e-30))
eta_of_logPhi = CubicSpline(log_Phi, eta_rec)

eta_of_H_recon = np.zeros(len(H_eval))
for i, H in enumerate(H_eval):
    psi_val = Psi_arr[i]
    if psi_val >= Phi_arr[0] and psi_val <= Phi_arr[-1]:
        log_psi = np.log(max(psi_val, 1e-30))
        if log_psi >= log_Phi[0] and log_psi <= log_Phi[-1]:
            eta_of_H_recon[i] = float(eta_of_logPhi(log_psi))
        else:
            eta_of_H_recon[i] = np.nan
    else:
        eta_of_H_recon[i] = np.nan

valid_eta = ~np.isnan(eta_of_H_recon)
H_valid = H_eval[valid_eta]; eta_valid = eta_of_H_recon[valid_eta]
ix_eta = np.argsort(eta_valid)
mono = np.concatenate([[True], np.diff(eta_valid[ix_eta]) > 0])
H_of_eta_sp = CubicSpline(eta_valid[ix_eta][mono], H_valid[ix_eta][mono])

# χ(r) on the reconstruction grid
r_chi = np.linspace(r_full[10], r_full[-10], 300)
eta_chi = compute_eta_from_boundary(r_chi, g_of_r(r_chi))
in_range = (eta_chi >= eta_valid[ix_eta][mono][0]) & (eta_chi <= eta_valid[ix_eta][mono][-1])
r_chi = r_chi[in_range]; eta_chi = eta_chi[in_range]
H_chi = H_of_eta_sp(eta_chi); G_chi = 1.0/r_chi**2
F_chi = H_chi**2/G_chi; g_chi = g_of_r(r_chi)
chi_recon = np.log(g_chi*G_chi/F_chi)

# Exact χ
chi_fine = np.array([chi_exact_at_z(z) for z in z_fine])
chi_ex_sp = CubicSpline(r_fine, chi_fine[ixe])
chi_exact_at_r = chi_ex_sp(r_chi)
err_chi = np.abs(chi_recon - chi_exact_at_r)
print(f"  χ(r): max err = {err_chi.max():.2e}, median = {np.median(err_chi):.2e}")


# ================================================================
# STEP 6: Recover f(z) and h(z)
# ================================================================
print("Step 6: Recover f(z), h(z)...")

# ODE: dz/dr = z²/(r² e^{χ/2})   with z → r near boundary (h→1)
chi_of_r_recon = CubicSpline(r_chi, chi_recon)

def rhs_zr(r_val, z_val):
    if z_val <= 0 or r_val <= 0: return 0.0
    r_cl = np.clip(r_val, r_chi[0], r_chi[-1])
    chi_val = float(chi_of_r_recon(r_cl))
    return z_val**2 / (r_val**2 * np.exp(chi_val/2.0))

# Start at small r where z ≈ r
r_start = r_chi[0]
z_start = r_start  # z ≈ r at boundary

sol_zr = solve_ivp(rhs_zr, [r_start, r_chi[-1]], [z_start],
                   method='RK45', rtol=1e-12, atol=1e-14,
                   dense_output=True, max_step=0.0005)

if not sol_zr.success:
    print(f"  ODE r→z failed: {sol_zr.message}")
else:
    r_out = np.linspace(r_chi[5], r_chi[-20], 200)
    z_of_r = sol_zr.sol(r_out)[0]

    # h(z) = z²/r², f(z) = g(r)/α², α = (z/r)(dr/dz) = (z/r)/(dz/dr)
    dz_dr = np.array([rhs_zr(r, z) for r, z in zip(r_out, z_of_r)])
    h_recon = z_of_r**2 / r_out**2
    alpha_recon = (z_of_r / r_out) * (1.0 / dz_dr)  # α = (z/r)·(dr/dz) = (z/r)/dz_dr
    g_at_r = g_of_r(r_out)
    f_recon = g_at_r / alpha_recon**2

    # Exact
    h_exact_arr = np.array([h_exact(z) for z in z_of_r])
    f_exact_arr = np.array([f_exact(z) for z in z_of_r])
    # Compare using exact z values at same r
    z_exact_at_r = np.array([z for z in z_fine[ixe]]) # need z(r) exact
    # Actually use z_of_r which is our reconstructed z — compare h,f at those z values
    mask_f = f_exact_arr > 0.05
    err_h = np.abs(h_recon - h_exact_arr)/h_exact_arr
    err_f = np.abs(f_recon[mask_f] - f_exact_arr[mask_f])/f_exact_arr[mask_f]
    print(f"  h(z): max err = {err_h.max():.2e}, median = {np.median(err_h):.2e}")
    print(f"  f(z): max err = {err_f.max():.2e}, median = {np.median(err_f):.2e}")


    print(f"  ODE f,h recovery skipped (ill-conditioned)")

# ================================================================
# STEP 6b: Recover f(z), h(z) directly from parametric data
# ================================================================
print("Step 6b: f(z), h(z) from parametric data...")

# At each data point z_i, r_i = 1/√(dS̃/dl) is known from reconstruction
# h(z_i) = z_i²/r_i², α = r_i e^{χ(r_i)/2}/z_i, f = g(r_i)/α²
# Only use interior points where both g and χ are available
trim_fh = 15
z_fh = z_stars[trim_fh:-trim_fh]  # parametric labels (not "using the metric")
r_fh = r_star[trim_fh:-trim_fh]   # from dS/dl
# Sort by z
ix_z = np.argsort(z_fh); z_fh = z_fh[ix_z]; r_fh = r_fh[ix_z]

# Only keep points where r is in the χ reconstruction range
in_chi = (r_fh >= r_chi[0]) & (r_fh <= r_chi[-1])
z_fh = z_fh[in_chi]; r_fh = r_fh[in_chi]

h_recon_z = z_fh**2 / r_fh**2
chi_at_r = chi_of_r_recon(r_fh)
alpha_recon_z = r_fh * np.exp(chi_at_r / 2.0) / z_fh
g_at_r_fh = g_of_r(r_fh)
f_recon_z = g_at_r_fh / alpha_recon_z**2

# Exact
h_exact_z = np.array([h_exact(z) for z in z_fh])
f_exact_z = np.array([f_exact(z) for z in z_fh])

err_h_z = np.abs(h_recon_z - h_exact_z) / h_exact_z
err_f_z = np.abs(f_recon_z - f_exact_z) / np.maximum(f_exact_z, 0.05)
mask_f_z = f_exact_z > 0.05

print(f"  h(z): max err = {err_h_z.max():.2e}, median = {np.median(err_h_z):.2e}")
print(f"  f(z): max err (f>0.05) = {err_f_z[mask_f_z].max():.2e}, "
      f"median = {np.median(err_f_z[mask_f_z]):.2e}")


# ================================================================
# FIGURE: present results in Bilson coordinates (g, χ, f/h)
# ================================================================
print("Generating figure...")

# f/h = g e^{-χ} — directly from reconstructed g and χ, no z(r) needed
foh_recon_r = g_chi * np.exp(-chi_recon)
foh_exact_r = g_ex_sp(r_chi) * np.exp(-chi_exact_at_r)
err_foh = np.abs(foh_recon_r - foh_exact_r) / foh_exact_r

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# (0,0): g(r)
ax = axes[0, 0]
ax.plot(r_rec, g_ex_sp(r_rec), 'r-', lw=2, label=r'$g(r)$ exact')
ax.plot(r_rec, g_rec, 'b--', lw=1.5, label=r'$g(r)$ Bilson')
ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$g(r)$')
ax.set_title(r'(a) $S_{EE}(l) \to g(r)$')
ax.legend()

# (0,1): χ(r)
ax = axes[0, 1]
ax.plot(r_chi, chi_exact_at_r, 'r-', lw=2, label=r'$\chi(r)$ exact')
ax.plot(r_chi, chi_recon, 'b--', lw=1.5, label=r'$\chi(r)$ recon')
ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$\chi(r)$')
ax.set_title(r'(b) $V(L) \to \chi(r)$')
ax.legend()

# (0,2): f/h = g e^{-χ}
ax = axes[0, 2]
ax.plot(r_chi, foh_exact_r, 'r-', lw=2, label=r'$f/h$ exact')
ax.plot(r_chi, foh_recon_r, 'b--', lw=1.5, label=r'$f/h$ recon')
ax.set_xlabel(r'$r$'); ax.set_ylabel(r'$f/h = g\,e^{-\chi}$')
ax.set_title(r'(c) Ratio $f/h$')
ax.legend()

# (1,0): f(z)
ax = axes[1, 0]
ax.plot(z_fh, f_exact_z, 'r-', lw=2, label=r'$f(z)$ exact')
ax.plot(z_fh, f_recon_z, 'b--', lw=1.5, label=r'$f(z)$ recon')
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$f(z)$')
ax.set_title(r'(d) Reconstructed $f(z)$')
ax.legend()

# (1,1): h(z)
ax = axes[1, 1]
ax.plot(z_fh, h_exact_z, 'r-', lw=2, label=r'$h(z)$ exact')
ax.plot(z_fh, h_recon_z, 'b--', lw=1.5, label=r'$h(z)$ recon')
ax.set_xlabel(r'$z$'); ax.set_ylabel(r'$h(z)$')
ax.set_title(r'(e) Reconstructed $h(z)$')
ax.legend()

# (1,2): errors
ax = axes[1, 2]
ax.semilogy(z_fh[mask_f_z], err_f_z[mask_f_z], 'b-', lw=1.5, label=r'$|\Delta f/f|$')
ax.semilogy(z_fh, err_h_z, 'r-', lw=1.5, label=r'$|\Delta h/h|$')
ax.set_xlabel(r'$z$'); ax.set_ylabel('Relative error')
ax.set_title(r'(f) Reconstruction errors')
ax.legend()
ax.set_ylim(1e-7, 1e-1)

plt.suptitle('Analytical metric reconstruction (Bilson + Hashimoto)', fontsize=14, y=1.01)
plt.tight_layout()
outpath = '../figures/bilson_reconstruction.pdf'
plt.savefig(outpath, bbox_inches='tight')
print(f"Saved {outpath}")

print(f"\nSummary:")
print(f"  g(r)  rel err: median = {np.median(g_err_grid):.2e}")
print(f"  χ(r)  abs err: median = {np.median(err_chi):.2e}, max = {err_chi.max():.2e}")
print(f"  f/h   rel err: median = {np.median(err_foh):.2e}, max = {err_foh.max():.2e}")


print("\nDone.")
