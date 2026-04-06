# ASSERT_CONVENTION: natural_units=natural, metric_signature=mostly_minus,
#   coordinate_system=fefferman_graham_z, z_h=1, d=4, L=1
# ASSERT_CONVENTION: f(z)=1-z^4, float64 for area
# ASSERT_CONVENTION: area = V_2 * int (1/z^3) sqrt(1 + z'^2/f(z)) dx
"""
Noise robustness study for the inverse problem.

Corrupts clean S_EE(l) data with multiplicative Gaussian noise at three
levels (0.1%, 1%, 5%) and tests whether the alternating ANN optimization
can still recover f(z) = 1 - z^4.

Noise model:  S_noisy(l) = S_clean(l) * (1 + sigma * N(0,1))
where sigma in {0.001, 0.01, 0.05}.

Random seed for noise: 123 (reproducibility).
Training seed: 42 (same as clean run).

References:
  - Filev, arXiv:2506.20115, Section 4
"""
import os
import sys
import time
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import interp1d

# --- Project imports ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import make_half_grid, load_ode_benchmark, Z_H
from ann_forward import EPSILON, X_S_FRAC
from ann_inverse import LModel, VModel, rt_area_V

rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# =============================================================================
# Training loop (adapted from ann_inverse.train_inverse)
# =============================================================================
def train_inverse_noisy(S_data_l, S_data_A, noise_sigma, n_epochs=500000,
                        lr_L=1e-4, lr_V=5e-4, n_grid=4000,
                        print_every=2000, checkpoint_every=50000,
                        seed=42, logger=None):
    """
    Run alternating optimization with noisy S_EE data.

    Parameters
    ----------
    S_data_l : np.ndarray
        Strip widths l_i.
    S_data_A : np.ndarray
        Noisy entanglement entropy values S_noisy(l_i).
    noise_sigma : float
        Noise level (for logging only; noise already applied to S_data_A).
    n_epochs : int
        Total training epochs.
    lr_L, lr_V : float
        Learning rates for L-model and V-model.
    n_grid : int
        Number of grid points for area integration.
    print_every : int
        Print progress every N epochs.
    checkpoint_every : int
        Save checkpoint every N epochs.
    seed : int
        Random seed for training.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    l_model : LModel
        Trained surface network z(x,l).
    v_model : VModel
        Trained blackening factor network f(z).
    loss_hist : list
        Loss history.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tag = f"{noise_sigma:.3f}"
    ckpt_dir = f"data/inverse_noise/checkpoints_{tag}"
    os.makedirs(ckpt_dir, exist_ok=True)

    l_min, l_max = float(S_data_l.min()), float(S_data_l.max())
    S_interp = interp1d(S_data_l, S_data_A, kind='cubic',
                        fill_value='extrapolate')

    l_model = LModel(hidden=16, depth=2)
    v_model = VModel(z_h=Z_H, hidden=16, depth=2)

    opt_L = torch.optim.Adam(l_model.parameters(), lr=lr_L)
    opt_V = torch.optim.Adam(v_model.parameters(), lr=lr_V)

    n_L = sum(p.numel() for p in l_model.parameters())
    n_V = sum(p.numel() for p in v_model.parameters())
    logger.info(f"  L-model: {n_L} params, V-model: {n_V} params")
    logger.info(f"  lr_L={lr_L}, lr_V={lr_V}, epochs={n_epochs}")
    logger.info(f"  Noise level sigma={noise_sigma}")

    loss_hist = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        l_val = float(np.random.uniform(l_min, l_max))
        S_target = float(S_interp(l_val))

        x_np = make_half_grid(l_val, n_grid, x_s_frac=X_S_FRAC)
        A_reg = rt_area_V(l_model, v_model, x_np, l_val)

        if epoch % 2 == 0:
            # Update V (data loss)
            opt_V.zero_grad()
            loss = 100.0 * (A_reg - S_target) ** 2
            loss.backward()
            opt_V.step()
        else:
            # Update L (minimize area)
            opt_L.zero_grad()
            loss = A_reg
            loss.backward()
            opt_L.step()

        loss_hist.append(loss.item())

        if epoch % print_every == 0 or epoch == 1:
            z_test = torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9],
                                  dtype=torch.float64)
            f_test = v_model(z_test).detach().numpy()
            f_exact = 1.0 - z_test.numpy() ** 4
            elapsed = time.time() - t0
            rate = epoch / elapsed
            logger.info(
                f"  ep {epoch:7d}/{n_epochs}  l={l_val:.3f}  "
                f"A_reg={A_reg.item():+.6f}  S_tgt={S_target:+.6f}  "
                f"|res|={abs(A_reg.item()-S_target):.4f}  "
                f"f(.3)={f_test[1]:.3f}({f_exact[1]:.3f}) "
                f"f(.5)={f_test[2]:.3f}({f_exact[2]:.3f}) "
                f"f(.7)={f_test[3]:.3f}({f_exact[3]:.3f}) "
                f"f(.9)={f_test[4]:.3f}({f_exact[4]:.3f})  "
                f"[{elapsed/60:.1f}min {rate:.0f}ep/s]"
            )

        if epoch % checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'l_model': l_model.state_dict(),
                'v_model': v_model.state_dict(),
            }, f"{ckpt_dir}/ep{epoch:06d}.pt")
            logger.info(f"  >> Checkpoint saved: ep {epoch}")

    t_train = time.time() - t0
    logger.info(f"  Training done: {t_train/60:.1f} min")
    return l_model, v_model, loss_hist


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_v_model(v_model, z_eval):
    """Evaluate learned f(z) on z_eval array."""
    z_t = torch.tensor(z_eval, dtype=torch.float64)
    with torch.no_grad():
        return v_model(z_t).numpy()


# =============================================================================
# Main
# =============================================================================
def main():
    # --- Load benchmark data ---
    ode = load_ode_benchmark("data/ode_benchmark.npz")
    l_c = ode['l_c']
    l_to_zs = interp1d(ode['l_of_zstar'], ode['z_star_grid'],
                        kind='cubic', fill_value='extrapolate')
    zs_to_Areg = interp1d(ode['z_star_grid'], ode['A_reg_of_zstar'],
                           kind='cubic', fill_value='extrapolate')

    # Clean S_EE data: same as the clean inverse run
    l_data = np.linspace(0.15, 0.75 * l_c, 50)
    A_clean = np.array([float(zs_to_Areg(float(l_to_zs(lv)))) for lv in l_data])

    # Evaluation grid for f(z)
    z_eval = np.linspace(0, Z_H, 200)
    f_exact = 1.0 - z_eval ** 4

    os.makedirs("data/inverse_noise", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Noise levels
    sigmas = [0.001, 0.01, 0.05]
    noise_seed = 123

    results = {}

    print("=" * 70)
    print("NOISE ROBUSTNESS STUDY: inverse problem f(z) from S_EE(l)")
    print(f"Noise model: S_noisy = S_clean * (1 + sigma * N(0,1))")
    print(f"Noise seed: {noise_seed}")
    print(f"Noise levels: {sigmas}")
    print(f"l range: [{l_data[0]:.3f}, {l_data[-1]:.3f}], {len(l_data)} points")
    print(f"Training: 500k epochs, lr_L=1e-4, lr_V=5e-4, hidden=16, depth=2")
    print("=" * 70)

    for sigma in sigmas:
        tag = f"{sigma:.3f}"
        out_path = f"data/inverse_noise/results_{tag}.npz"

        # Skip if already completed
        if os.path.exists(out_path):
            print(f"\n  sigma={sigma}: already done ({out_path}), loading...")
            d = np.load(out_path)
            results[sigma] = {
                'f_learned': d['f_learned'],
                'max_rel': float(d['max_rel_err']),
                'mean_rel': float(d['mean_rel_err']),
                'max_abs': float(d['max_abs_err']),
                'mean_abs': float(d['mean_abs_err']),
            }
            continue

        print(f"\n{'='*70}")
        print(f"  NOISE LEVEL: sigma = {sigma}")
        print(f"{'='*70}")

        # --- Set up per-sigma logger ---
        log_file = f"inverse_noise_{tag}_training.log"
        logger = logging.getLogger(f"noise_{tag}")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(fh)
        logger.addHandler(sh)

        logger.info(f"Noise robustness: sigma={sigma}")

        # --- Generate noisy data ---
        rng = np.random.RandomState(noise_seed)
        noise = rng.randn(len(A_clean))
        A_noisy = A_clean * (1.0 + sigma * noise)

        logger.info(f"  Clean S_EE range: [{A_clean.min():.4f}, {A_clean.max():.4f}]")
        logger.info(f"  Noisy S_EE range: [{A_noisy.min():.4f}, {A_noisy.max():.4f}]")
        max_perturbation = np.max(np.abs(A_noisy - A_clean) / np.abs(A_clean))
        logger.info(f"  Max relative perturbation: {max_perturbation:.6f}")

        # --- Train ---
        l_model, v_model, loss_hist = train_inverse_noisy(
            l_data, A_noisy,
            noise_sigma=sigma,
            n_epochs=500000,
            lr_L=1e-4,
            lr_V=5e-4,
            n_grid=4000,
            print_every=2000,
            checkpoint_every=50000,
            seed=42,
            logger=logger,
        )

        # --- Evaluate f(z) ---
        f_learned = evaluate_v_model(v_model, z_eval)

        # Error metrics (exclude near-horizon z > 0.95 where f -> 0)
        mask = z_eval < 0.95
        abs_err = np.abs(f_learned[mask] - f_exact[mask])
        rel_err = abs_err / np.maximum(np.abs(f_exact[mask]), 1e-10)
        max_rel = rel_err.max()
        mean_rel = rel_err.mean()
        max_abs = abs_err.max()
        mean_abs = abs_err.mean()

        logger.info(f"  f(z) error (z<0.95): max_rel={max_rel:.4f}, "
                     f"mean_rel={mean_rel:.4f}, max_abs={max_abs:.4f}, "
                     f"mean_abs={mean_abs:.4f}")

        # --- Save results ---
        out_path = f"data/inverse_noise/results_{tag}.npz"
        np.savez(out_path,
                 z=z_eval, f_learned=f_learned, f_exact=f_exact,
                 l_data=l_data, A_clean=A_clean, A_noisy=A_noisy,
                 sigma=sigma, noise_seed=noise_seed,
                 max_rel_err=max_rel, mean_rel_err=mean_rel,
                 max_abs_err=max_abs, mean_abs_err=mean_abs,
                 loss_hist=np.array(loss_hist))
        logger.info(f"  Saved: {out_path}")

        results[sigma] = {
            'f_learned': f_learned.copy(),
            'max_rel': max_rel,
            'mean_rel': mean_rel,
            'max_abs': max_abs,
            'mean_abs': mean_abs,
        }

        # Clean up handlers
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 70)
    print("  NOISE ROBUSTNESS SUMMARY")
    print("=" * 70)
    print(f"{'Noise sigma':>12s}  {'Max |rel err|':>14s}  {'Mean |rel err|':>14s}  "
          f"{'Max |abs err|':>14s}  {'Mean |abs err|':>14s}")
    print("-" * 74)
    for sigma in sigmas:
        r = results[sigma]
        print(f"{sigma:12.3f}  {r['max_rel']:14.4f}  {r['mean_rel']:14.4f}  "
              f"{r['max_abs']:14.4f}  {r['mean_abs']:14.4f}")
    print("=" * 70)

    # =========================================================================
    # Publication figure: inverse_noise_robustness.pdf
    # =========================================================================
    colors = {0.001: '#2166ac', 0.01: '#f4a582', 0.05: '#b2182b'}
    labels = {0.001: r'$\sigma = 0.1\%$', 0.01: r'$\sigma = 1\%$',
              0.05: r'$\sigma = 5\%$'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left panel: f(z) recovered ---
    ax1.plot(z_eval, f_exact, 'k-', lw=2.0, label=r'Exact: $f(z)=1-z^4$',
             zorder=10)
    for sigma in sigmas:
        r = results[sigma]
        lbl = (f"{labels[sigma]}: max err {r['max_rel']:.1%}, "
               f"mean {r['mean_rel']:.1%}")
        ax1.plot(z_eval, r['f_learned'], '--', color=colors[sigma],
                 lw=1.5, label=lbl)
    ax1.set_xlabel(r'$z / z_h$')
    ax1.set_ylabel(r'$f(z)$')
    ax1.set_title('Recovered blackening factor')
    ax1.legend(fontsize=8, loc='lower left')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    # --- Right panel: relative error ---
    for sigma in sigmas:
        r = results[sigma]
        rel = np.abs(r['f_learned'] - f_exact) / np.maximum(np.abs(f_exact), 1e-10)
        # Clip near-horizon where f_exact -> 0
        rel[z_eval > 0.95] = np.nan
        ax2.semilogy(z_eval, rel, '-', color=colors[sigma], lw=1.5,
                     label=labels[sigma])
    ax2.set_xlabel(r'$z / z_h$')
    ax2.set_ylabel(r'$|f_{\mathrm{learned}} - f_{\mathrm{exact}}| / |f_{\mathrm{exact}}|$')
    ax2.set_title('Relative error in $f(z)$')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 0.95)
    ax2.grid(alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig("figures/inverse_noise_robustness.pdf")
    plt.close()
    print("\nSaved: figures/inverse_noise_robustness.pdf")
    print("DONE.")


if __name__ == "__main__":
    main()
