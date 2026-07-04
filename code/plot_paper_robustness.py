"""
Paper figures for the revised Section 5.3 "Robustness of the reconstruction"
(referee point 3).

Figure 1 (a_seed_convergence.pdf):
    Training trajectories of the boundary derivative a = f'(0) = h'(0) for
    the three-network S_EE + V(L) reconstruction: 10 independent seeds at
    5e4 epochs, 4 of them extended to 5e5 epochs. All lock onto a = 1.5 --
    the counterpart of the EE-only drift figure (a_drift_no_wl.pdf).

Figure 2 (noise_robustness_wl.pdf), three panels:
    (a) recovered f(z) for the five sigma=5% noise realizations vs exact;
    (b) recovered h(z) likewise;
    (c) recovered a vs sigma with 95% t-confidence intervals
        (sigma = 0, 1%, 5%, all at the fixed 3e5-epoch budget).

Outputs are written to figures/ and copied to paper/figures/.
"""
import os
import glob
import shutil
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SS = os.path.join(BASE, 'data', 'seed_study')
FIGS = os.path.join(BASE, 'figures')
PFIGS = os.path.join(BASE, 'paper', 'figures')
TARGET_EP = 300000     # fixed budget for the noise comparison

# =========================================================================
# Figure 1: a(epoch) trajectories, 10 seeds (50k) + 4 extended (500k)
# =========================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.2))

short = sorted(glob.glob(os.path.join(SS, 'wl_50k', 'wl_seed*.npz')))
extended = sorted(glob.glob(os.path.join(SS, 'wl_500k', 'wl_seed*.npz')))
ext_seeds = {int(np.load(f)['seed']) for f in extended}

for f in short:
    d = np.load(f)
    if int(d['seed']) in ext_seeds:
        continue                      # drawn below with their full history
    ax.plot(d['a_hist_ep'] / 1e3, d['a_hist'], '-', color='#7fa8d9',
            lw=0.9, alpha=0.9)
colors = plt.cm.viridis(np.linspace(0.05, 0.75, len(extended)))
for f, c in zip(extended, colors):
    d = np.load(f)
    ax.plot(d['a_hist_ep'] / 1e3, d['a_hist'], '-', color=c, lw=1.2,
            label=f"seed {int(d['seed'])} (extended)")
ax.axhline(1.5, color='red', ls='--', lw=1.2, label=r'Exact $a=\frac{3}{2}Q=1.5$')
ax.set_xscale('log')
ax.set_xlim(0.5, 600)
ax.set_ylim(0.95, 1.62)
ax.set_xlabel(r'Epoch ($\times 10^3$)')
ax.set_ylabel(r"$a = f'(0) = h'(0)$")
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3, which='both')
out1 = os.path.join(FIGS, 'a_seed_convergence.pdf')
fig.savefig(out1)
plt.close()
print('saved', out1)

# =========================================================================
# Figure 2: noise robustness (f, h overlays at sigma=5%; a vs sigma with CI)
# =========================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

noise5 = sorted(glob.glob(os.path.join(SS, 'wl_noise500k', 'wl_sig0.050_ts*.npz')))
d0 = np.load(noise5[0])
z = d0['z_eval']
ax1.plot(z, d0['f_exact'], 'k-', lw=2.0, label='Exact', zorder=10)
ax2.plot(z, d0['h_exact'], 'k-', lw=2.0, label='Exact', zorder=10)
ncols = plt.cm.plasma(np.linspace(0.05, 0.8, len(noise5)))
for f, c in zip(noise5, ncols):
    d = np.load(f)
    ax1.plot(z, d['f_final'], '--', color=c, lw=1.2)
    ax2.plot(z, d['h_final'], '--', color=c, lw=1.2)
ax1.set_xlabel(r'$z$'); ax1.set_ylabel(r'$f(z)$')
ax1.set_title(r'(a) $f(z)$, five realizations at $\sigma = 5\%$')
ax1.legend(); ax1.grid(alpha=0.3)
ax2.set_xlabel(r'$z$'); ax2.set_ylabel(r'$h(z)$')
ax2.set_title(r'(b) $h(z)$, five realizations at $\sigma = 5\%$')
ax2.legend(); ax2.grid(alpha=0.3)

# panel (c): a vs sigma with 95% CI
def a_clean_at(target):
    vals = []
    for f in sorted(glob.glob(os.path.join(SS, 'wl_500k', 'wl_seed*.npz'))):
        d = np.load(f)
        i = np.argmin(np.abs(d['a_hist_ep'] - target))
        vals.append(float(d['a_hist'][i]))
    return np.array(vals)

def a_noise(sig):
    return np.array([float(np.load(f)['a_final']) for f in sorted(
        glob.glob(os.path.join(SS, 'wl_noise500k', f'wl_sig{sig:.3f}_ts*.npz')))])

sig_pts, means, halfw = [], [], []
for sig, vals in [(0.0, a_clean_at(TARGET_EP)), (0.01, a_noise(0.01)),
                  (0.05, a_noise(0.05))]:
    n = len(vals)
    m, sd = vals.mean(), vals.std(ddof=1)
    hw = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    sig_pts.append(sig * 100); means.append(m); halfw.append(hw)
    print(f'sigma={sig*100:g}%: a = {m:.4f} +/- {hw:.4f} (95% CI, n={n})')

ax3.errorbar(sig_pts, means, yerr=halfw, fmt='o', color='#2166ac',
             capsize=5, lw=1.6, markersize=6,
             label=r'$a$ (mean $\pm$ 95% CI)')
ax3.axhline(1.5, color='red', ls='--', lw=1.2, label=r'Exact $a = 1.5$')
ax3.set_xlabel(r'Noise level $\sigma$ (%)')
ax3.set_ylabel(r"$a = f'(0) = h'(0)$")
ax3.set_title(r'(c) Recovered $a$ vs noise level')
ax3.set_xlim(-0.5, 5.8)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(alpha=0.3)

plt.tight_layout()
out2 = os.path.join(FIGS, 'noise_robustness_wl.pdf')
fig.savefig(out2)
plt.close()
print('saved', out2)

for f in (out1, out2):
    shutil.copy(f, PFIGS)
    print('copied to', os.path.join(PFIGS, os.path.basename(f)))
