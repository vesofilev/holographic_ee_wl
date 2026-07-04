"""
Regenerate the two h=1-dependent paper figures from the upgraded (structured
encoding) five-seed ensemble:

  figures/inverse_metric.pdf    Sec 5.1: recovered f(z) + learned S_EE(l)
                                (representative run: seed 5)
  figures/accuracy_summary.pdf  Sec 6: left panel unchanged (forward results),
                                right panel = inverse f(z) errors from the
                                new representative run

Old versions are backed up as figures/<name>_v1_original.pdf.
Both outputs are copied to paper/figures/.
"""
import os
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(BASE, 'figures')
PFIGS = os.path.join(BASE, 'paper', 'figures')
REP = os.path.join(BASE, 'data', 'seed_study', 'h1v2',
                   'h1v2_structured0_w100_s05.npz')   # representative run

for name in ('inverse_metric.pdf', 'accuracy_summary.pdf'):
    src = os.path.join(PFIGS, name)
    bak = os.path.join(FIGS, name.replace('.pdf', '_v1_original.pdf'))
    if os.path.exists(src) and not os.path.exists(bak):
        shutil.copy(src, bak)
        print('backed up', name, '->', bak)

d = np.load(REP)
z, f_l, f_e = d['z_eval'], d['f_final'], d['f_exact']
l_eval, A_ann, A_ref = d['l_eval'], d['A_ann'], d['A_ref']

# ---------------------------------------------------------------------
# Figure 1: inverse_metric.pdf (same layout as the original)
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(z, f_e, 'r-', lw=1.5, label=r'Exact: $f(z) = 1 - z^4$')
ax1.plot(z, f_l, 'b--', lw=1.5, label='Learned $f(z)$')
ax1.set_xlabel(r'$z / z_h$')
ax1.set_ylabel(r'$f(z)$')
ax1.set_title('Recovered blackening factor')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(l_eval, A_ref, 'r-', lw=1.2, label='Input data (ODE)')
ax2.plot(l_eval, A_ann, 'b--', lw=1.0, label='Learned $S_{EE}(l)$')
ax2.set_xlabel(r'$l / z_h$')
ax2.set_ylabel(r'$A_{\mathrm{reg,half}}$')
ax2.set_title('Entanglement entropy: learned vs data')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
out1 = os.path.join(FIGS, 'inverse_metric.pdf')
fig.savefig(out1)
plt.close()
print('saved', out1)

# ---------------------------------------------------------------------
# Figure 2: accuracy_summary.pdf (left panel identical to plot_comparison.py)
# ---------------------------------------------------------------------
single_metrics = []
for i in range(4):
    with open(os.path.join(BASE, f"data/ann_single_l/metrics_l{i}.json")) as f:
        single_metrics.append(json.load(f))
with open(os.path.join(BASE, "data/conditional/results.json")) as f:
    cond_results = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

l_single = [m['l'] for m in single_metrics]
err_single = [m['rel_area_error'] for m in single_metrics]
ax1.semilogy(l_single, err_single, 'rs', ms=8, label='Single-$l$ ANN')
l_cond = [r['l'] for r in cond_results]
err_cond = [r['rel_error'] for r in cond_results]
ax1.semilogy(l_cond, err_cond, 'b-', lw=1.0, label='Conditional ANN')
ax1.axhline(y=0.01, color='gray', ls='--', lw=0.8, label='1% target')
ax1.axhline(y=0.001, color='gray', ls=':', lw=0.8, label='0.1% target')
ax1.set_xlabel(r'$l / z_h$')
ax1.set_ylabel('Relative error in $A_{\\mathrm{reg}}$')
ax1.set_title(r'$S_{EE}(l)$ accuracy')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_ylim(1e-5, 1)

mask = z < 0.95
rel_err = np.abs(f_l[mask] - f_e[mask]) / np.maximum(np.abs(f_e[mask]), 1e-10)
abs_err_full = np.abs(f_l - f_e)
ax2.semilogy(z[mask], rel_err, 'b-', lw=1.0, label='Relative error ($z<0.95$)')
ax2.semilogy(z, abs_err_full, 'g--', lw=0.8, label='Absolute error')
ax2.axhline(y=0.01, color='gray', ls=':', lw=0.8, label='1%')
ax2.axhline(y=0.001, color='gray', ls='--', lw=0.8, label='0.1%')
ax2.set_xlabel(r'$z / z_h$')
ax2.set_ylabel('Error in $f(z)$')
ax2.set_title('Inverse problem: $f(z)$ accuracy')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_ylim(1e-6, 0.1)

plt.tight_layout()
out2 = os.path.join(FIGS, 'accuracy_summary.pdf')
fig.savefig(out2)
plt.close()
print('saved', out2)

for f in (out1, out2):
    shutil.copy(f, PFIGS)
    print('copied to paper/figures:', os.path.basename(f))
