"""
Reconstruct χ(r) via the separable integral approach.

Φ(η) = ∫_{-∞}^{η} dη'/√G(η') = ∫_{-∞}^{η} r(η') dη'
Ψ(H) = ∫_H^{∞} σ(H') dH'

Match Φ(η) = Ψ(H) to get η(H), hence H(η), F=H²/G, χ=log(gG/F).
"""
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import CubicSpline
from generate_data_hp import r_of_z, g_exact_at_z, chi_exact_at_z, f_exact, h_exact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_eta_from_boundary(r_arr, g_arr):
    """η(r) = ln(r) + ∫_0^r [1/(r'√g) - 1/r'] dr'"""
    reg = (1.0/np.sqrt(g_arr) - 1.0) / r_arr
    return np.log(r_arr) + cumulative_trapezoid(reg, r_arr, initial=0)


# ================================================================
# Load high-precision data
# ================================================================
d = np.load('bilson_data_hp.npz')
l, A, L, V = d['l'], d['A'], d['L'], d['V']


# ================================================================
# Bilson g(r) [condensed — already validated to 10^{-9}]
# ================================================================
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

print(f"g(r): r=[{r_full[0]:.4f}, {r_full[-1]:.4f}]")


# ================================================================
# η(r) and G(η) — boundary side
# ================================================================
eta_rec = compute_eta_from_boundary(r_full, g_full)
G_rec = 1.0/r_full**2
G_of_eta_sp = CubicSpline(eta_rec, G_rec)
r_of_eta_sp = CubicSpline(eta_rec, r_full)

print(f"η: [{eta_rec[0]:.2f}, {eta_rec[-1]:.2f}]")


# ================================================================
# Φ(η) = ∫_{-∞}^{η} r(η') dη'  [since 1/√G = r]
# ================================================================
# === Subtracted approach: work with deviations from pure AdS ===
# δΦ(η) = Φ(η) - e^η = ∫_{-∞}^η [r(η') - e^{η'}] dη'
# Integrand: r(η) - e^η → 0 at boundary, so integral converges well
r_ads = np.exp(eta_rec)  # pure AdS: r = e^η
delta_r = r_full - r_ads  # deviation from AdS

# Tail: ∫_{-∞}^{η_min} [r - e^η] dη ≈ g₂ r_min³/12 (from expansion)
g2_val = bdy_poly(0)
r_min_phi = r_full[0]
delta_Phi_tail = g2_val * r_min_phi**3 / 12.0
delta_Phi_arr = delta_Phi_tail + cumulative_trapezoid(delta_r, eta_rec, initial=0)

# Full Φ = e^η + δΦ
Phi_arr = np.exp(eta_rec) + delta_Phi_arr

print(f"  δΦ range: [{delta_Phi_arr.min():.6e}, {delta_Phi_arr.max():.6e}]")
print(f"Φ: [{Phi_arr[0]:.6f}, {Phi_arr[-1]:.6f}]")


# ================================================================
# σ(H) from Abel inversion [condensed — already validated]
# ================================================================
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

print(f"σ(H): H=[{Hg[0]:.2f}, {Hg[-1]:.2f}]")


# ================================================================
# Ψ(H) = ∫_H^{∞} σ(H') dH'
# ================================================================
# For H > h0_data_max: σ ≈ 1/(2H^{3/2}) (pure AdS)
# ∫_H^∞ dH'/(2H'^{3/2}) = 1/√H
# For H in data range: numerical integration + tail correction

H_eval = np.exp(np.linspace(np.log(Hg[0]), np.log(Hg[-1]), 500))
H_max_data = Hg[-1]

# === Subtracted Ψ: δΨ(H) = Ψ(H) - 1/√H = ∫_H^∞ [σ(H') - 1/(2H'^{3/2})] dH' ===
def delta_sigma(Hv):
    return sigma_of_H(Hv) - 0.5 * Hv**(-1.5)

# Tail for δσ: fit δσ · H^{5/2} near H_max to get leading coefficient
H_fit_mask = (Hg >= 0.5*H_max_data) & (Hg <= H_max_data)
dsig_vals = (sigma_arr[H_fit_mask] - 0.5*Hg[H_fit_mask]**(-1.5)) * Hg[H_fit_mask]**2.5
C_fit = np.mean(dsig_vals)
delta_tail = 2.0*C_fit / (3.0*H_max_data**1.5)
print(f"  δσ·H^(5/2) ≈ {C_fit:.6f}, δtail = {delta_tail:.2e}")

delta_Psi_arr = np.zeros(len(H_eval))
for i, H in enumerate(H_eval):
    val, _ = quad(delta_sigma, H, H_max_data, epsabs=1e-15, epsrel=1e-15, limit=2000)
    delta_Psi_arr[i] = val + delta_tail

# Full Ψ = 1/√H + δΨ
Psi_arr = 1.0/np.sqrt(H_eval) + delta_Psi_arr

print(f"  δΨ range: [{delta_Psi_arr.min():.6e}, {delta_Psi_arr.max():.6e}]")
print(f"Ψ: [{Psi_arr.min():.6f}, {Psi_arr.max():.6f}]")


# ================================================================
# Match: Φ(η) = Ψ(H) → η(H)
# ================================================================
# For each H, Ψ(H) is known. Find η such that Φ(η) = Ψ(H).
# Φ is monotonically increasing (integrand r > 0), so invertible.
# Match Φ(η) = Ψ(H) using log-space inversion (Φ computed with subtraction)
# The Φ values now benefit from the subtracted computation (less roundoff)
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
print(f"η(H) valid: {valid_eta.sum()}/{len(H_eval)}")


# ================================================================
# Exact η(H) for comparison
# ================================================================
z_pts = np.linspace(0.005, 0.93, 2000)
r_ex = np.array([r_of_z(z) for z in z_pts])
g_ex = np.array([g_exact_at_z(z) for z in z_pts])
chi_ex = np.array([chi_exact_at_z(z) for z in z_pts])
ixe = np.argsort(r_ex); r_ex = r_ex[ixe]; g_ex = g_ex[ixe]; chi_ex = chi_ex[ixe]

H_exact = np.sqrt(g_ex*np.exp(-chi_ex))/r_ex**2
eta_exact = compute_eta_from_boundary(r_ex, g_ex)


# ================================================================
# Plot η vs H
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.semilogx(H_exact, eta_exact, 'r-', lw=2, label=r'exact $\eta(H)$')
ax.semilogx(H_eval[valid_eta], eta_of_H_recon[valid_eta], 'b--', lw=1.5,
            label=r'recon $\eta(H)$')
ax.set_xlabel('$H$'); ax.set_ylabel(r'$\eta$')
ax.set_title(r'$\eta(H)$ — separable integral')
ax.legend()

# Zoom on common range
ax = axes[1]
H_common_mask = (H_exact >= H_eval[valid_eta].min()) & (H_exact <= H_eval[valid_eta].max())
ax.semilogx(H_exact[H_common_mask], eta_exact[H_common_mask], 'r-', lw=2, label='exact')
ax.semilogx(H_eval[valid_eta], eta_of_H_recon[valid_eta], 'b--', lw=1.5, label='recon')
ax.set_xlabel('$H$'); ax.set_ylabel(r'$\eta$')
ax.set_title(r'$\eta(H)$ zoom')
ax.legend()

# Error: interpolate exact to same H values and compare
H_both = H_eval[valid_eta]
eta_rec_vals = eta_of_H_recon[valid_eta]
# Exact η(H): need to interpolate. H_exact is not monotonic? It should be decreasing.
# Sort by H
ixH = np.argsort(H_exact)
eta_of_H_exact_sp = CubicSpline(H_exact[ixH], eta_exact[ixH])
H_in_range = (H_both >= H_exact[ixH][0]) & (H_both <= H_exact[ixH][-1])
if H_in_range.sum() > 0:
    eta_exact_at_H = eta_of_H_exact_sp(H_both[H_in_range])
    eta_rec_at_H = eta_rec_vals[H_in_range]
    err_eta = eta_rec_at_H - eta_exact_at_H

    ax = axes[2]
    ax.semilogx(H_both[H_in_range], err_eta, 'k-', lw=1.5)
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('$H$'); ax.set_ylabel(r'$\eta_{\rm recon} - \eta_{\rm exact}$')
    ax.set_title(r'$\eta(H)$ error')

    print(f"\nη(H) error: max={np.abs(err_eta).max():.2e}, "
          f"mean={np.abs(err_eta).mean():.2e}, median={np.median(np.abs(err_eta)):.2e}")

plt.tight_layout()
plt.savefig('../figures/eta_vs_H.pdf', bbox_inches='tight')
print("Saved figures/eta_vs_H.pdf")


# ================================================================
# Reconstruct χ(r)
# ================================================================
# From η(H): invert to H(η), then F = H²/G, χ = log(gG/F)

# Build H(η) from the reconstruction
# η_of_H_recon is η as function of H (H_eval, eta_of_H_recon)
# Need to invert: H as function of η
# η decreases with H (more negative η = larger H = closer to boundary)
H_valid = H_eval[valid_eta]
eta_valid = eta_of_H_recon[valid_eta]

# Sort by η (increasing)
ix_eta = np.argsort(eta_valid)
eta_sorted = eta_valid[ix_eta]
H_sorted = H_valid[ix_eta]

# Ensure monotonicity
mono = np.concatenate([[True], np.diff(eta_sorted) > 0])
eta_sorted = eta_sorted[mono]
H_sorted = H_sorted[mono]

H_of_eta_recon = CubicSpline(eta_sorted, H_sorted)

# Evaluate on the r grid where we have g(r)
# Use the interior of the Bilson grid (avoid edges)
r_chi = np.linspace(r_full[10], r_full[-10], 300)
eta_chi = compute_eta_from_boundary(r_chi, g_of_r(r_chi))

# Only use points where η is in the reconstruction range
in_range = (eta_chi >= eta_sorted[0]) & (eta_chi <= eta_sorted[-1])
r_chi = r_chi[in_range]
eta_chi = eta_chi[in_range]

G_chi = 1.0 / r_chi**2
H_chi = H_of_eta_recon(eta_chi)
F_chi = H_chi**2 / G_chi
g_chi = g_of_r(r_chi)

chi_recon = np.log(g_chi * G_chi / F_chi)

# Exact χ(r)
chi_exact_arr = np.array([chi_exact_at_z(z) for z in z_pts])
chi_ex_sorted = chi_exact_arr[ixe]  # sorted by r
chi_of_r_exact = CubicSpline(r_ex, chi_ex_sorted)
chi_exact_at_r = chi_of_r_exact(r_chi)

err_chi = np.abs(chi_recon - chi_exact_at_r)
print(f"\nχ(r) reconstruction:")
print(f"  max err  = {err_chi.max():.2e}")
print(f"  mean err = {err_chi.mean():.2e}")
print(f"  median   = {np.median(err_chi):.2e}")
print(f"  recon: [{chi_recon.min():.6f}, {chi_recon.max():.6f}]")
print(f"  exact: [{chi_exact_at_r.min():.6f}, {chi_exact_at_r.max():.6f}]")

# ================================================================
# Plot χ(r)
# ================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes2[0]
ax.plot(r_chi, chi_exact_at_r, 'r-', lw=2, label=r'$\chi(r)$ exact')
ax.plot(r_chi, chi_recon, 'b--', lw=1.5, label=r'$\chi(r)$ recon')
ax.set_xlabel('$r$'); ax.set_ylabel(r'$\chi(r)$')
ax.set_title(r'$\chi(r)$ reconstruction')
ax.legend()

ax = axes2[1]
ax.plot(r_chi, chi_exact_at_r, 'r-', lw=2, label='exact')
ax.plot(r_chi, chi_recon, 'b--', lw=1.5, label='recon')
ax.set_xlabel('$r$'); ax.set_ylabel(r'$\chi(r)$')
ax.set_title(r'$\chi(r)$ zoom')
ax.legend()

ax = axes2[2]
ax.plot(r_chi, err_chi, 'k-', lw=1.5)
ax.axhline(0, color='gray', ls='--', lw=0.5)
ax.set_xlabel('$r$'); ax.set_ylabel(r'$|\chi_{\rm recon} - \chi_{\rm exact}|$')
ax.set_title(r'$\chi(r)$ absolute error')

plt.tight_layout()
plt.savefig('../figures/chi_reconstruction.pdf', bbox_inches='tight')
print("Saved figures/chi_reconstruction.pdf")


# ================================================================
# Plot F(η) reconstructed vs exact
# ================================================================
# Exact F(η) = g(r) e^{-χ(r)} / r²
F_exact_at_r = g_ex * np.exp(-chi_ex_sorted) / r_ex**2
eta_exact_full = compute_eta_from_boundary(r_ex, g_ex)

# Reconstructed F(η) = H(η)² / G(η) = H(η)² r(η)²
F_recon_at_eta = H_chi**2 / G_chi  # = H² r²

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes3[0]
ax.semilogy(eta_exact_full, F_exact_at_r, 'r-', lw=2, label='$F(\\eta)$ exact')
ax.semilogy(eta_chi, F_recon_at_eta, 'b--', lw=1.5, label='$F(\\eta)$ recon')
ax.set_xlabel('$\\eta$'); ax.set_ylabel('$F(\\eta)$')
ax.set_title('$F(\\eta)$ comparison')
ax.legend()

# Zoom on common range
eta_min_F = max(eta_chi.min(), eta_exact_full.min())
eta_max_F = min(eta_chi.max(), eta_exact_full.max())
ax = axes3[1]
m_ex = (eta_exact_full >= eta_min_F) & (eta_exact_full <= eta_max_F)
m_rec = (eta_chi >= eta_min_F) & (eta_chi <= eta_max_F)
ax.semilogy(eta_exact_full[m_ex], F_exact_at_r[m_ex], 'r-', lw=2, label='exact')
ax.semilogy(eta_chi[m_rec], F_recon_at_eta[m_rec], 'b--', lw=1.5, label='recon')
ax.set_xlabel('$\\eta$'); ax.set_ylabel('$F(\\eta)$')
ax.set_title('$F(\\eta)$ zoom')
ax.legend()

# Relative error on common η grid
eta_F_common = np.linspace(eta_min_F + 0.1, eta_max_F - 0.1, 200)
F_ex_sp = CubicSpline(eta_exact_full, F_exact_at_r)
F_rec_sp = CubicSpline(eta_chi, F_recon_at_eta)
ok_F = (eta_F_common >= eta_exact_full[0]) & (eta_F_common <= eta_exact_full[-1]) & \
       (eta_F_common >= eta_chi[0]) & (eta_F_common <= eta_chi[-1])
eta_F_ok = eta_F_common[ok_F]
F_ex_vals = F_ex_sp(eta_F_ok)
F_rec_vals = F_rec_sp(eta_F_ok)
rel_err_F = (F_rec_vals - F_ex_vals) / F_ex_vals

ax = axes3[2]
ax.plot(eta_F_ok, rel_err_F, 'k-', lw=1.5)
ax.axhline(0, color='gray', ls='--', lw=0.5)
ax.set_xlabel('$\\eta$'); ax.set_ylabel('$(F_{\\rm recon} - F_{\\rm exact})/F_{\\rm exact}$')
ax.set_title('$F(\\eta)$ relative error')

plt.tight_layout()
plt.savefig('../figures/F_eta_diagnostic.pdf', bbox_inches='tight')
print(f"Saved figures/F_eta_diagnostic.pdf")
print(f"F rel error: max={np.abs(rel_err_F).max():.2e}, mean={np.abs(rel_err_F).mean():.2e}, "
      f"median={np.median(np.abs(rel_err_F)):.2e}")
