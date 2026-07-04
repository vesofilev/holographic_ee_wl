"""
Aggregate the referee-point-3 seed study into tables and paper figures.

Inputs (whatever exists is aggregated; missing files are reported):
  data/seed_study/wl/wl_seed*.npz         Study C: 3-network WL, 10 seeds
  data/seed_study/noise/noise_sig*.npz    Study B: 5 new realizations / sigma
  data/inverse_noise/results_*.npz        Study B: original realization (seed 42)
  data/seed_study/h1/h1_seed*.npz         Study A: 9 new clean seeds
  data/inverse/results.npz                Study A: original clean run (seed 42)
  data/seed_study/drift/drift_seed*.npz   Study D: EE-only drift, 4 seeds
  inverse_gr_d3_run_v6h(.cont).log        Study D: original drift run (parsed)

Outputs:
  data/seed_study/SUMMARY.md              markdown tables + LaTeX-ready numbers
  figures/a_seeds_wl_vs_drift.pdf         killer figure: flat direction vs WL lock
  figures/wl_seed_spread.pdf              Study C: f/h error spread across seeds
  figures/noise_seed_stats.pdf            Study B: error vs sigma with spread
  figures/h1_seed_spread.pdf              Study A: f(z) error spread across seeds

Usage: python code/aggregate_seed_study.py
"""
import os
import re
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(CODE_DIR)
SS = os.path.join(BASE, 'data', 'seed_study')
FIGS = os.path.join(BASE, 'figures')

# Directory suffix for reduced-epoch test runs (e.g. --suffix _50k)
SUFFIX = ''
if '--suffix' in sys.argv:
    SUFFIX = sys.argv[sys.argv.index('--suffix') + 1]

lines_md = []          # accumulated SUMMARY.md content


def md(s=""):
    lines_md.append(s)
    print(s)


def stats(vals):
    v = np.asarray(vals, dtype=float)
    return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, v.min(), v.max()


# =========================================================================
# Study C: WL three-network
# =========================================================================
def aggregate_wl():
    files = sorted(glob.glob(os.path.join(SS, 'wl' + SUFFIX, 'wl_seed*.npz')))
    if not files:
        md("## Study C (WL 3-network): NO DATA YET\n")
        return None
    runs = [dict(np.load(f)) for f in files]
    md(f"## Study C: three-network WL reconstruction ({len(runs)} seeds)\n")
    md("| seed | a_final | max df/f | mean df/f | max dh/h | mean dh/h | h(zh) | min |")
    md("|---|---|---|---|---|---|---|---|")
    for r in runs:
        md(f"| {int(r['seed'])} | {float(r['a_final']):.4f} "
           f"| {float(r['f_max_err']):.4f} | {float(r['f_mean_err']):.4f} "
           f"| {float(r['h_max_err']):.4f} | {float(r['h_mean_err']):.4f} "
           f"| {float(r['h_zh_final']):.4f} | {float(r['wall_min']):.0f} |")
    md("")
    for key, label in [('a_final', 'a (exact 1.5)'),
                       ('f_max_err', 'max |df/f|'), ('f_mean_err', 'mean |df/f|'),
                       ('h_max_err', 'max |dh/h|'), ('h_mean_err', 'mean |dh/h|'),
                       ('h_zh_final', 'h(z_h) (exact 2.8284)')]:
        m, s, lo, hi = stats([float(r[key]) for r in runs])
        md(f"- **{label}**: {m:.5f} +/- {s:.5f}  (range [{lo:.5f}, {hi:.5f}])")
    md("")
    return runs


# =========================================================================
# Study D: EE-only drift
# =========================================================================
def parse_drift_log(path):
    """(epoch, a) pairs from a v6h-style training log."""
    epochs, a_vals = [], []
    ep = None
    if not os.path.exists(path):
        return np.array([]), np.array([])
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


def aggregate_drift():
    files = sorted(glob.glob(os.path.join(SS, 'drift' + SUFFIX, 'drift_seed*.npz')))
    runs = [dict(np.load(f)) for f in files]
    if runs:
        md(f"## Study D: EE-only drift ({len(runs)} seeds, no Wilson loop)\n")
        md("| seed | a_final | a range over run | h(zh) |")
        md("|---|---|---|---|")
        for r in runs:
            ah = r['a_hist']
            md(f"| {int(r['seed'])} | {float(r['a_final']):.4f} "
               f"| [{ah.min():.4f}, {ah.max():.4f}] "
               f"| {float(r['h_zh_final']):.4f} |")
        md("")
    else:
        md("## Study D (EE-only drift): NO DATA YET\n")
    return runs


# =========================================================================
# Killer figure: a(epoch) with WL (locked) vs without WL (wandering)
# =========================================================================
def fig_a_seeds(wl_runs, drift_runs):
    if not wl_runs and not drift_runs:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)

    # Left: EE-only drift
    cmap_d = plt.cm.autumn(np.linspace(0.05, 0.75, max(len(drift_runs), 1)))
    for i, r in enumerate(drift_runs):
        ax1.plot(r['a_hist_ep'] / 1e3, r['a_hist'], '-', lw=1.0,
                 color=cmap_d[i], label=f"seed {int(r['seed'])}")
    ep0, a0 = parse_drift_log(os.path.join(BASE, 'inverse_gr_d3_run_v6h.log'))
    if len(ep0):
        ax1.plot(ep0 / 1e3, a0, '-', lw=1.0, color='0.45',
                 label='paper run')
    ax1.axhline(1.5, color='red', ls='--', lw=1.0)
    ax1.set_xlabel(r'Epoch ($\times 10^3$)')
    ax1.set_ylabel(r"$a = f'(0) = h'(0)$")
    ax1.set_title(r'$S_{EE}$ only (flat direction)')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(alpha=0.3)

    # Right: with Wilson loop
    cmap_w = plt.cm.winter(np.linspace(0, 0.9, max(len(wl_runs), 1)))
    for i, r in enumerate(wl_runs):
        ax2.plot(r['a_hist_ep'] / 1e3, r['a_hist'], '-', lw=0.9,
                 color=cmap_w[i], alpha=0.8)
    ax2.axhline(1.5, color='red', ls='--', lw=1.0,
                label=r'Exact $a = \frac{3}{2}Q = 1.5$')
    ax2.set_xlabel(r'Epoch ($\times 10^3$)')
    ax2.set_title(f'$S_{{EE}}$ + Wilson loop ({len(wl_runs)} seeds)')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGS, f'a_seeds_wl_vs_drift{SUFFIX}.pdf')
    fig.savefig(out)
    plt.close()
    md(f"Figure saved: {out}\n")


def fig_wl_spread(wl_runs):
    if not wl_runs:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    cmap = plt.cm.winter(np.linspace(0, 0.9, len(wl_runs)))
    for i, r in enumerate(wl_runs):
        z = r['z_eval']
        f_rel = np.abs(r['f_final'] - r['f_exact']) / np.abs(r['f_exact'])
        h_rel = np.abs(r['h_final'] - r['h_exact']) / np.abs(r['h_exact'])
        ax1.semilogy(z, f_rel, '-', lw=0.8, color=cmap[i], alpha=0.8)
        ax2.semilogy(z, h_rel, '-', lw=0.8, color=cmap[i], alpha=0.8)
    for ax, lab in [(ax1, r'$|\Delta f/f|$'), (ax2, r'$|\Delta h/h|$')]:
        ax.set_xlabel(r'$z/z_h$')
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, which='both')
    ax1.set_title(f'Blackening factor error ({len(wl_runs)} seeds)')
    ax2.set_title(f'Warp factor error ({len(wl_runs)} seeds)')
    plt.tight_layout()
    out = os.path.join(FIGS, f'wl_seed_spread{SUFFIX}.pdf')
    fig.savefig(out)
    plt.close()
    md(f"Figure saved: {out}\n")


# =========================================================================
# Study B: noise
# =========================================================================
def aggregate_noise():
    sigmas = [0.001, 0.01, 0.05]
    table = {}
    for sig in sigmas:
        entries = []
        orig = os.path.join(BASE, 'data', 'inverse_noise', f'results_{sig:.3f}.npz')
        if os.path.exists(orig):
            d = np.load(orig)
            entries.append({'label': 'ts42/ns123 (paper)',
                            'max': float(d['max_rel_err']),
                            'mean': float(d['mean_rel_err']),
                            'f': d['f_learned'], 'z': d['z']})
        for f in sorted(glob.glob(os.path.join(SS, 'noise' + SUFFIX,
                                               f'noise_sig{sig:.3f}_ts*.npz'))):
            d = np.load(f)
            entries.append({'label': f"ts{int(d['seed'])}/ns{int(d['noise_seed'])}",
                            'max': float(d['max_rel_err']),
                            'mean': float(d['mean_rel_err']),
                            'f': d['f_final'], 'z': d['z_eval']})
        table[sig] = entries

    md("## Study B: noise robustness (h=1), realizations per level\n")
    for sig in sigmas:
        entries = table[sig]
        md(f"### sigma = {sig*100:g}%  ({len(entries)} realizations)")
        if not entries:
            continue
        md("| realization | max df/f | mean df/f |")
        md("|---|---|---|")
        for e in entries:
            md(f"| {e['label']} | {e['max']:.4f} | {e['mean']:.4f} |")
        m1, s1, lo1, hi1 = stats([e['max'] for e in entries])
        m2, s2, lo2, hi2 = stats([e['mean'] for e in entries])
        md(f"- max df/f: {m1:.4f} +/- {s1:.4f} (range [{lo1:.4f}, {hi1:.4f}]); "
           f"median {np.median([e['max'] for e in entries]):.4f}")
        md(f"- mean df/f: {m2:.4f} +/- {s2:.4f} (range [{lo2:.4f}, {hi2:.4f}]); "
           f"median {np.median([e['mean'] for e in entries]):.4f}")
        md("")
    return table


def fig_noise(table):
    entries_all = [e for v in table.values() for e in v]
    if not entries_all:
        return
    sigmas = sorted(table.keys())
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for stat_key, color, label in [('max', '#b2182b', r'max $|\Delta f/f|$'),
                                   ('mean', '#2166ac', r'mean $|\Delta f/f|$')]:
        med, lo_v, hi_v = [], [], []
        for sig in sigmas:
            vals = np.array([e[stat_key] for e in table[sig]])
            if len(vals) == 0:
                med.append(np.nan); lo_v.append(np.nan); hi_v.append(np.nan)
                continue
            med.append(np.median(vals))
            lo_v.append(vals.min()); hi_v.append(vals.max())
            ax.scatter([sig * 100] * len(vals), vals, s=18, color=color,
                       alpha=0.45, zorder=3)
        ax.plot(np.array(sigmas) * 100, med, 'o-', color=color, lw=1.6,
                label=label + ' (median)', zorder=4)
        ax.fill_between(np.array(sigmas) * 100, lo_v, hi_v, color=color,
                        alpha=0.12, zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Input noise $\sigma$ (%)')
    ax.set_ylabel(r'Error in recovered $f(z)$  ($z < 0.95\,z_h$)')
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    plt.tight_layout()
    out = os.path.join(FIGS, f'noise_seed_stats{SUFFIX}.pdf')
    fig.savefig(out)
    plt.close()
    md(f"Figure saved: {out}\n")


# =========================================================================
# Study A: clean h=1
# =========================================================================
def aggregate_h1():
    entries = []
    orig = os.path.join(BASE, 'data', 'inverse', 'results.npz')
    if os.path.exists(orig):
        d = np.load(orig)
        z, fl, fe = d['z'], d['f_learned'], d['f_exact']
        mask = z < 0.95
        rel = np.abs(fl[mask] - fe[mask]) / np.maximum(np.abs(fe[mask]), 1e-10)
        entries.append({'label': 'seed 42 (paper)', 'max': rel.max(),
                        'mean': rel.mean(), 'f': fl, 'z': z})
    for f in sorted(glob.glob(os.path.join(SS, 'h1' + SUFFIX, 'h1_seed*.npz'))):
        d = np.load(f)
        entries.append({'label': f"seed {int(d['seed'])}",
                        'max': float(d['max_rel_err']),
                        'mean': float(d['mean_rel_err']),
                        'f': d['f_final'], 'z': d['z_eval']})
    md(f"## Study A: clean h=1 inverse ({len(entries)} seeds)\n")
    if entries:
        md("| run | max df/f | mean df/f |")
        md("|---|---|---|")
        for e in entries:
            md(f"| {e['label']} | {e['max']:.4f} | {e['mean']:.4f} |")
        m1, s1, lo1, hi1 = stats([e['max'] for e in entries])
        m2, s2, lo2, hi2 = stats([e['mean'] for e in entries])
        md(f"- max df/f: {m1:.4f} +/- {s1:.4f} (range [{lo1:.4f}, {hi1:.4f}])")
        md(f"- mean df/f: {m2:.4f} +/- {s2:.4f} (range [{lo2:.4f}, {hi2:.4f}])")
        md("")
    return entries


def fig_h1(entries):
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cmap = plt.cm.winter(np.linspace(0, 0.9, len(entries)))
    for i, e in enumerate(entries):
        z, fl = e['z'], e['f']
        fe = 1.0 - z ** 4
        rel = np.abs(fl - fe) / np.maximum(np.abs(fe), 1e-10)
        rel[z > 0.95] = np.nan
        ax.semilogy(z, rel, '-', lw=0.8, color=cmap[i], alpha=0.85)
    ax.set_xlabel(r'$z/z_h$')
    ax.set_ylabel(r'$|\Delta f/f|$')
    ax.set_title(f'Recovered $f(z)$: relative error, {len(entries)} seeds')
    ax.grid(alpha=0.3, which='both')
    ax.set_xlim(0, 0.95)
    plt.tight_layout()
    out = os.path.join(FIGS, f'h1_seed_spread{SUFFIX}.pdf')
    fig.savefig(out)
    plt.close()
    md(f"Figure saved: {out}\n")


def main():
    md("# Seed study summary (referee point 3)\n")
    wl_runs = aggregate_wl()
    drift_runs = aggregate_drift()
    noise_table = aggregate_noise()
    h1_entries = aggregate_h1()

    os.makedirs(FIGS, exist_ok=True)
    fig_a_seeds(wl_runs or [], drift_runs or [])
    fig_wl_spread(wl_runs or [])
    fig_noise(noise_table or {})
    fig_h1(h1_entries or [])

    out_md = os.path.join(SS, f'SUMMARY{SUFFIX}.md')
    with open(out_md, 'w') as fh:
        fh.write("\n".join(lines_md) + "\n")
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
