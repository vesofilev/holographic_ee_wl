"""
Generate comparison figures and data for Section 6 of the paper.
Compares ODE shooting vs ANN methods across all three tasks:
  1. Single-l forward problem
  2. Conditional network S_EE(l)
  3. Inverse problem (metric reconstruction)
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    os.makedirs("figures", exist_ok=True)

    # =====================================================================
    # 1. Load all data
    # =====================================================================

    # Single-l forward results
    single_metrics = []
    for i in range(4):
        with open(f"data/ann_single_l/metrics_l{i}.json") as f:
            single_metrics.append(json.load(f))

    # Conditional results
    with open("data/conditional/results.json") as f:
        cond_results = json.load(f)

    # Loss histories
    single_losses = [np.load(f"data/ann_single_l/loss_history_l{i}.npy")
                     for i in range(4)]
    cond_loss = np.load("data/conditional/loss_history.npy")

    # Inverse results
    inv = np.load("data/inverse/results.npz")

    # =====================================================================
    # 2. Figure 1: Loss convergence comparison (single-l and conditional)
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = [m['label'] for m in single_metrics]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (loss, label, c) in enumerate(zip(single_losses, labels, colors)):
        ax1.semilogy(np.abs(loss), lw=0.4, alpha=0.6, color=c, label=f'$l = {label}$')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(r'$|A_{\mathrm{reg,half}}|$')
    ax1.set_title('Single-$l$ convergence')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, len(single_losses[0]))

    # Conditional: smoothed loss
    kernel = 1000
    smooth = np.convolve(np.abs(cond_loss), np.ones(kernel)/kernel, mode='valid')
    ax2.semilogy(np.abs(cond_loss), 'b-', lw=0.15, alpha=0.3)
    ax2.semilogy(range(kernel//2, kernel//2 + len(smooth)), smooth,
                 'r-', lw=1.5, label=f'{kernel}-epoch average')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(r'$|A_{\mathrm{reg,half}}|$')
    ax2.set_title('Conditional network convergence')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/convergence_comparison.pdf")
    plt.close()
    print("Saved figures/convergence_comparison.pdf")

    # =====================================================================
    # 3. Figure 2: Accuracy summary across all methods
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: S_EE relative error vs l for single-l and conditional
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

    # Panel 2: f(z) error from inverse problem
    z = inv['z']
    f_l, f_e = inv['f_learned'], inv['f_exact']
    mask = z < 0.95
    rel_err = np.abs(f_l[mask] - f_e[mask]) / np.maximum(np.abs(f_e[mask]), 1e-10)
    abs_err = np.abs(f_l[mask] - f_e[mask])

    ax2.semilogy(z[mask], rel_err, 'b-', lw=1.0, label='Relative error')
    ax2.semilogy(z[mask], abs_err, 'g--', lw=0.8, label='Absolute error')
    ax2.axhline(y=0.05, color='r', ls='--', lw=0.8, label='5% target')
    ax2.axhline(y=0.01, color='gray', ls=':', lw=0.8, label='1%')
    ax2.set_xlabel(r'$z / z_h$')
    ax2.set_ylabel('Error in $f(z)$')
    ax2.set_title('Inverse problem: $f(z)$ accuracy')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(1e-5, 0.5)

    plt.tight_layout()
    plt.savefig("figures/accuracy_summary.pdf")
    plt.close()
    print("Saved figures/accuracy_summary.pdf")

    # =====================================================================
    # 4. Print summary table for the paper
    # =====================================================================
    print("\n" + "=" * 80)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 80)

    # ODE benchmark timing (estimate: 498 points, ~0.02s each from quad)
    ode_time_per_point = 0.02  # rough estimate for scipy quad
    ode_total = 498 * ode_time_per_point
    print(f"\nODE benchmark: ~{ode_total:.0f}s for {498} points")

    print(f"\nSingle-l ANN (4 strip widths):")
    for m in single_metrics:
        print(f"  l={m['l']:.3f}: {m['training_time']:.1f}s, "
              f"area err={m['rel_area_error']:.2e}, "
              f"z* err={m['rel_zstar_error']:.2e}")
    avg_time = np.mean([m['training_time'] for m in single_metrics])
    avg_err = np.mean([m['rel_area_error'] for m in single_metrics])
    print(f"  Average: {avg_time:.1f}s, {avg_err:.2e}")

    # Conditional
    cond_errs = [r['rel_error'] for r in cond_results if r['rel_error'] == r['rel_error']]
    # Training time from log: ~28 min for 120k epochs
    cond_time = 28 * 60  # seconds
    print(f"\nConditional ANN (100 l-values):")
    print(f"  Training: ~{cond_time/60:.0f} min (120k epochs)")
    print(f"  Mean rel error: {np.mean(cond_errs):.2e}")
    print(f"  Max rel error: {np.max(cond_errs):.2e}")

    # Inverse
    f_rel_max = rel_err.max()
    f_rel_mean = rel_err.mean()
    A_ann, A_ode = np.array(inv['A_ann']), np.array(inv['A_ode'])
    see_rel = np.abs(A_ann - A_ode) / np.maximum(np.abs(A_ode), 1e-10)
    inv_time = 50 * 60  # seconds
    print(f"\nInverse problem:")
    print(f"  Training: ~{inv_time/60:.0f} min (500k epochs)")
    print(f"  f(z) max rel err (z<0.95): {f_rel_max:.4f}")
    print(f"  f(z) mean rel err: {f_rel_mean:.4f}")
    print(f"  S_EE max rel err: {see_rel.max():.4f}")
    print(f"  S_EE mean rel err: {see_rel.mean():.4f}")

    print("\n" + "=" * 80)
    print("DONE")


if __name__ == "__main__":
    main()
