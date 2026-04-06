#!/usr/bin/env python3
"""Plot the drift of boundary derivative a = f'(0) = h'(0) in the
inverse Gubser-Rocha training WITHOUT Wilson loop data.

Parses the training logs:
  - inverse_gr_d3_run_v6h.log       (epochs 1-500k)
  - inverse_gr_d3_run_v6h_cont.log  (epochs 502k-774k, continuation)

Produces: figures/a_drift_no_wl.pdf
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'text.usetex': False,
})

def parse_log(path):
    """Extract (epoch, a) pairs from a training log."""
    epochs, a_vals = [], []
    ep = None
    with open(path) as f:
        for line in f:
            m_ep = re.search(r'ep\s+(\d+)/\d+', line)
            m_a = re.search(r'a=([\d.]+)\(exact', line)
            if m_ep:
                ep = int(m_ep.group(1))
            if m_a and ep is not None:
                a_vals.append(float(m_a.group(1)))
                epochs.append(ep)
    return np.array(epochs), np.array(a_vals)

# --- Parse logs ---
ep1, a1 = parse_log('../inverse_gr_d3_run_v6h.log')
ep2, a2 = parse_log('../inverse_gr_d3_run_v6h_cont.log')

# Combine into single series
epochs = np.concatenate([ep1, ep2])
a_vals = np.concatenate([a1, a2])

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 3.5))

ax.plot(epochs / 1e3, a_vals, '-', color='C0', linewidth=1.2,
        label=r'$a$ (no Wilson loop)')
ax.axhline(y=1.5, color='red', linestyle='--', linewidth=1.0,
           label=r'Exact $a = \frac{3}{2}Q = 1.5$')

# Mark the boundary between original and continuation runs
ax.axvline(x=500, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
ax.text(505, 1.05, 'cont.', fontsize=8, color='gray', va='bottom')

ax.set_xlabel('Epoch ($\\times 10^3$)')
ax.set_ylabel('$a = f\'(0) = h\'(0)$')
ax.set_xlim(0, epochs[-1] / 1e3 + 10)
ax.set_ylim(0.95, 1.65)
ax.legend(loc='lower right')
ax.set_title(r'Boundary derivative drift without Wilson loop data')

plt.tight_layout()
plt.savefig('../figures/a_drift_no_wl.pdf', bbox_inches='tight')
print(f"Saved figures/a_drift_no_wl.pdf")
print(f"  Epochs: {epochs[0]} to {epochs[-1]}")
print(f"  a range: {a_vals[0]:.4f} to {a_vals[-1]:.4f}")
print(f"  Final drift from exact: {(a_vals[-1] - 1.5)/1.5 * 100:.1f}%")
