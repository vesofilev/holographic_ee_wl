# ASSERT_CONVENTION: AdS4 (d=3 boundary) Gubser-Rocha, Q=1
# ASSERT_CONVENTION: protocol identical to ann_inverse_gr_d3_wl.py (paper run)
"""
Seed study, Study C: one seed of the three-network WL reconstruction.

Lean replica of train_inverse_gr_d3_wl (ann_inverse_gr_d3_wl.py) with the
EXACT paper protocol -- same data, models, learning rates, 4-step alternating
cycle (V, L, V, W), 10000x data-loss weights, no thermodynamic penalty --
but without snapshot plots / periodic checkpoints, and with per-epoch-history
recording of the boundary derivative a and the metric errors.

Referee point (3): random-seed dependence / reproducibility statistics.

Usage:  python seed_study_wl.py --seed 3 [--epochs 500000]
Output: data/seed_study/wl/wl_seed03.npz (+ _models.pt)
"""
import os
import sys
import time
import argparse
import logging

# Configure root logger BEFORE importing project modules: their
# logging.basicConfig calls (which would truncate shared log files in CWD)
# then become no-ops.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('seed_study_wl')

import numpy as np
import torch
import torch.optim as optim
from scipy.interpolate import interp1d

torch.set_num_threads(1)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from utils import make_half_grid, Z_H
from ann_forward import X_S_FRAC
from ann_inverse import LModel
from ann_inverse_gr_d3 import VModelGR_d3, rt_area_V_gr_d3, D_BOUNDARY_D3
from ann_inverse_gr_d3_wl import WModel, wl_action_V_gr_d3
from ode_benchmark_gr_d3 import f_exact, h_exact

WL_CACHE = os.path.join(BASE_DIR, 'data', 'seed_study', 'wl_data_cache.npz')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'seed_study', 'wl')

Q = 1.0
A_HIST_STRIDE = 100
ERR_HIST_STRIDE = 5000


def load_training_data(sigma=0.0, noise_seed=0):
    """S_EE from the ODE benchmark (connected branch, l > 0.15) and
    Wilson loop data from the cache (z* in [0.15, 0.90], 150 pts) --
    identical to the __main__ block of ann_inverse_gr_d3_wl.py.

    If sigma > 0, multiplicative Gaussian noise is applied to the two
    OBSERVABLE arrays only -- S_EE(l) and V(L) -- with independent noise
    streams. The coordinates (l, L), the exact metric, and everything used
    for error evaluation stay clean:
        S_noisy = S_clean * (1 + sigma * xi_S),  xi_S ~ N(0,1)
        V_noisy = V_clean * (1 + sigma * xi_V),  xi_V ~ N(0,1)  (independent)
    """
    data = np.load(os.path.join(BASE_DIR, 'data', 'ode_benchmark_gr_d3.npz'))
    l_all = data['l_of_zstar']
    A_all = data['A_reg_of_zstar']
    mask = (l_all > 0.15) & (A_all < 0)
    S_data_l, S_data_A = l_all[mask], A_all[mask]

    if not os.path.exists(WL_CACHE):
        logger.info("WL data cache missing -- generating (one-off)...")
        from bilson_reconstruction_v2 import compute_L_WL, compute_Vreg_WL
        z_stars = np.linspace(0.15, 0.90, 150)
        L_arr = np.array([compute_L_WL(zs) for zs in z_stars])
        V_arr = np.array([compute_Vreg_WL(zs) for zs in z_stars])
        os.makedirs(os.path.dirname(WL_CACHE), exist_ok=True)
        np.savez(WL_CACHE, z_stars=z_stars, L=L_arr, V=V_arr, Q=Q)
        logger.info(f"  cached -> {WL_CACHE}")
    cache = np.load(WL_CACHE)
    V_data_L, V_data_V = np.array(cache['L']), np.array(cache['V'])
    S_data_A = np.array(S_data_A)

    if sigma > 0:
        # Independent noise streams for the two observables (reproducible).
        rng_S = np.random.RandomState(noise_seed)
        rng_V = np.random.RandomState(noise_seed + 100000)
        S_data_A = S_data_A * (1.0 + sigma * rng_S.randn(len(S_data_A)))
        V_data_V = V_data_V * (1.0 + sigma * rng_V.randn(len(V_data_V)))
        logger.info(f"  NOISE sigma={sigma} applied to S_EE and V(L) "
                    f"(noise_seed={noise_seed}); coordinates/metric clean")
    return S_data_l, S_data_A, V_data_L, V_data_V


def metric_errors(v_model, z_eval_t, f_ex, h_ex):
    with torch.no_grad():
        f_pred = v_model.forward_f(z_eval_t).numpy()
        h_pred = v_model.forward_h(z_eval_t).numpy()
    f_rel = np.abs(f_pred - f_ex) / np.abs(f_ex)
    h_rel = np.abs(h_pred - h_ex) / np.abs(h_ex)
    return f_pred, h_pred, f_rel, h_rel


def run(seed, n_epochs, lr_L=1e-4, lr_W=1e-4, lr_V=5e-4, n_grid=4000,
        outdir_suffix='', resume_from=None, ckpt_every=50000,
        sigma=0.0, noise_seed=0):
    out_dir = OUT_DIR + outdir_suffix
    S_data_l, S_data_A, V_data_L, V_data_V = load_training_data(sigma, noise_seed)
    base = (f"wl_sig{sigma:.3f}_ts{seed:02d}_ns{noise_seed}"
            if sigma > 0 else f"wl_seed{seed:02d}")

    l_min, l_max = float(S_data_l.min()), float(S_data_l.max())
    L_min, L_max = float(V_data_L.min()), float(V_data_L.max())
    S_interp = interp1d(S_data_l, S_data_A, kind='cubic', fill_value='extrapolate')
    V_interp = interp1d(V_data_L, V_data_V, kind='cubic', fill_value='extrapolate')
    s_target = (1.0 + Q) ** 1.5

    # Fresh path: seed BEFORE constructing models so the same seed reproduces
    # the same init (this ordering is the fix for the h1 seeding bug -- do NOT
    # move construction above this).
    if resume_from is None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    l_model = LModel(hidden=32, depth=2, d=D_BOUNDARY_D3)
    w_model = WModel(hidden=32, depth=2, d=D_BOUNDARY_D3)
    v_model = VModelGR_d3(z_h=Z_H, h_h=s_target, hidden=20, depth=2)

    opt_L = optim.Adam(l_model.parameters(), lr=lr_L)
    opt_W = optim.Adam(w_model.parameters(), lr=lr_W)
    opt_V = optim.Adam(v_model.parameters(), lr=lr_V)

    # History (continued across a resume)
    prev = {}
    start_epoch = 1
    if resume_from is not None:
        ck = torch.load(resume_from, weights_only=False)
        l_model.load_state_dict(ck['l_model'])
        w_model.load_state_dict(ck['w_model'])
        v_model.load_state_dict(ck['v_model'])
        start_epoch = int(ck['epoch']) + 1
        # Faithful resume restores the Adam moments; if the checkpoint predates
        # optimizer-saving (the original 50k runs), fall back to fresh Adam.
        if ck.get('opt_L') is not None:
            opt_L.load_state_dict(ck['opt_L'])
            opt_W.load_state_dict(ck['opt_W'])
            opt_V.load_state_dict(ck['opt_V'])
            opt_note = 'model+optimizer (faithful)'
        else:
            opt_note = ('model-only, FRESH Adam (checkpoint predates optimizer '
                        'saving; safe here because 50k state is already converged)')
        # Restore RNG streams if present, else reseed for fresh sampling.
        if ck.get('torch_rng') is not None:
            torch.set_rng_state(ck['torch_rng'])
            np.random.set_state(ck['np_rng'])
        else:
            torch.manual_seed(seed * 100003 + start_epoch)
            np.random.seed((seed * 100003 + start_epoch) % (2**31 - 1))
        # Carry forward the pre-resume history so trajectories are continuous.
        prev_npz = resume_from.replace('_models.pt', '.npz')
        if os.path.exists(prev_npz):
            p = np.load(prev_npz)
            for k in ('a_hist_ep', 'a_hist', 'err_ep', 'f_max_hist',
                      'f_mean_hist', 'h_max_hist', 'h_mean_hist'):
                if k in p.files:
                    prev[k] = list(p[k])
        logger.info(f"  Resumed from {resume_from} (epoch {start_epoch - 1}): {opt_note}")

    def _save_ckpt(ep):
        """Full checkpoint: models + optimizer moments + RNG -> faithful resume."""
        torch.save({'l_model': l_model.state_dict(),
                    'w_model': w_model.state_dict(),
                    'v_model': v_model.state_dict(),
                    'opt_L': opt_L.state_dict(),
                    'opt_W': opt_W.state_dict(),
                    'opt_V': opt_V.state_dict(),
                    'torch_rng': torch.get_rng_state(),
                    'np_rng': np.random.get_state(),
                    'seed': seed, 'sigma': sigma, 'noise_seed': noise_seed,
                    'epoch': ep},
                   os.path.join(out_dir, f"{base}_models.pt"))

    logger.info(f"seed={seed}  epochs={start_epoch}..{n_epochs}  "
                f"S_EE: {len(S_data_l)} pts l=[{l_min:.4f},{l_max:.4f}]  "
                f"V(L): {len(V_data_L)} pts L=[{L_min:.4f},{L_max:.4f}]")

    # Evaluation grid (matches paper metric: z in [0.01, 0.99])
    z_eval = np.linspace(0.01, 0.99, 200)
    z_eval_t = torch.tensor(z_eval, dtype=torch.float64)
    f_ex = np.array([f_exact(z, Q) for z in z_eval])
    h_ex = np.array([h_exact(z, Q) for z in z_eval])

    a_hist_ep = prev.get('a_hist_ep', [])
    a_hist = prev.get('a_hist', [])
    err_ep = prev.get('err_ep', [])
    f_max_hist = prev.get('f_max_hist', [])
    f_mean_hist = prev.get('f_mean_hist', [])
    h_max_hist = prev.get('h_max_hist', [])
    h_mean_hist = prev.get('h_mean_hist', [])

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for epoch in range(start_epoch, n_epochs + 1):
        step_type = epoch % 4  # 0: V-step, 1: L-step, 2: V-step, 3: W-step

        if step_type in (0, 2):
            opt_V.zero_grad()
            l_val = float(np.random.uniform(l_min, l_max))
            S_target = float(S_interp(l_val))
            x_np = make_half_grid(l_val, n_grid, x_s_frac=X_S_FRAC)
            A_reg = rt_area_V_gr_d3(l_model, v_model, x_np, l_val)
            see_loss = 10000.0 * (A_reg - S_target)**2

            L_val = float(np.random.uniform(L_min, L_max))
            V_target = float(V_interp(L_val))
            V_reg = wl_action_V_gr_d3(w_model, v_model, x_np, L_val)
            wl_loss = 10000.0 * (V_reg - V_target)**2

            loss = see_loss + wl_loss
            loss.backward()
            opt_V.step()

        elif step_type == 1:
            l_val = float(np.random.uniform(l_min, l_max))
            x_np = make_half_grid(l_val, n_grid, x_s_frac=X_S_FRAC)
            A_reg = rt_area_V_gr_d3(l_model, v_model, x_np, l_val)
            opt_L.zero_grad()
            loss = A_reg
            loss.backward()
            opt_L.step()

        elif step_type == 3:
            L_val = float(np.random.uniform(L_min, L_max))
            x_np = make_half_grid(L_val, n_grid, x_s_frac=X_S_FRAC)
            V_reg = wl_action_V_gr_d3(w_model, v_model, x_np, L_val)
            opt_W.zero_grad()
            loss = V_reg
            loss.backward()
            opt_W.step()

        if epoch % A_HIST_STRIDE == 0:
            a_hist_ep.append(epoch)
            a_hist.append(v_model.a.item())

        if epoch % ERR_HIST_STRIDE == 0:
            _, _, f_rel, h_rel = metric_errors(v_model, z_eval_t, f_ex, h_ex)
            err_ep.append(epoch)
            f_max_hist.append(f_rel.max()); f_mean_hist.append(f_rel.mean())
            h_max_hist.append(h_rel.max()); h_mean_hist.append(h_rel.mean())

        if epoch % 25000 == 0 or epoch == start_epoch:
            el = time.time() - t0
            rate = (epoch - start_epoch + 1) / el if el > 0 else 0
            a_val = v_model.a.item()
            fm = f_max_hist[-1] if f_max_hist else float('nan')
            hm = h_max_hist[-1] if h_max_hist else float('nan')
            logger.info(f"  ep {epoch:7d}/{n_epochs}  a={a_val:.4f}(1.5000)  "
                        f"|f err|={fm:.4f}  |h err|={hm:.4f}  "
                        f"[{el/60:.1f}min {rate:.0f}ep/s]")

        # Periodic full checkpoint (model+optimizer+RNG) for faithful resume.
        if epoch % ckpt_every == 0:
            _save_ckpt(epoch)

    wall_min = (time.time() - t0) / 60.0

    # --- Final evaluation ---
    f_final, h_final, f_rel, h_rel = metric_errors(v_model, z_eval_t, f_ex, h_ex)
    a_final = v_model.a.item()
    with torch.no_grad():
        h_zh = v_model.forward_h(torch.tensor([Z_H], dtype=torch.float64)).item()

    # Data-fit residuals on 20-point grids
    l_model.eval(); w_model.eval(); v_model.eval()
    l_fit = np.linspace(l_min, l_max, 20)
    L_fit = np.linspace(L_min, L_max, 20)
    A_fit, S_ref, V_fit, V_ref = [], [], [], []
    for lv in l_fit:
        x_np = make_half_grid(lv, 2000, x_s_frac=X_S_FRAC)
        A_fit.append(rt_area_V_gr_d3(l_model, v_model, x_np, lv).item())
        S_ref.append(float(S_interp(lv)))
    for Lv in L_fit:
        x_np = make_half_grid(Lv, 2000, x_s_frac=X_S_FRAC)
        V_fit.append(wl_action_V_gr_d3(w_model, v_model, x_np, Lv).item())
        V_ref.append(float(V_interp(Lv)))

    out_npz = os.path.join(out_dir, f"{base}.npz")
    np.savez(out_npz,
             seed=seed, sigma=sigma, noise_seed=noise_seed,
             Q=Q, n_epochs=n_epochs, wall_min=wall_min,
             lr_L=lr_L, lr_W=lr_W, lr_V=lr_V, n_grid=n_grid,
             a_hist_ep=np.array(a_hist_ep), a_hist=np.array(a_hist),
             err_ep=np.array(err_ep),
             f_max_hist=np.array(f_max_hist), f_mean_hist=np.array(f_mean_hist),
             h_max_hist=np.array(h_max_hist), h_mean_hist=np.array(h_mean_hist),
             z_eval=z_eval, f_final=f_final, h_final=h_final,
             f_exact=f_ex, h_exact=h_ex,
             a_final=a_final, h_zh_final=h_zh,
             f_max_err=f_rel.max(), f_mean_err=f_rel.mean(),
             h_max_err=h_rel.max(), h_mean_err=h_rel.mean(),
             l_fit=l_fit, A_fit=np.array(A_fit), S_ref=np.array(S_ref),
             L_fit=L_fit, V_fit=np.array(V_fit), V_ref=np.array(V_ref))
    _save_ckpt(n_epochs)

    logger.info(f"DONE seed={seed}: a={a_final:.4f}  "
                f"f: max={f_rel.max():.4f} mean={f_rel.mean():.4f}  "
                f"h: max={h_rel.max():.4f} mean={h_rel.mean():.4f}  "
                f"h(zh)={h_zh:.4f}({s_target:.4f})  [{wall_min:.1f} min]")
    logger.info(f"Saved: {out_npz}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=500000)
    ap.add_argument('--outdir-suffix', type=str, default='')
    ap.add_argument('--resume-from', type=str, default=None)
    ap.add_argument('--ckpt-every', type=int, default=50000)
    ap.add_argument('--sigma', type=float, default=0.0)
    ap.add_argument('--noise-seed', type=int, default=0)
    args = ap.parse_args()
    run(args.seed, args.epochs, outdir_suffix=args.outdir_suffix,
        resume_from=args.resume_from, ckpt_every=args.ckpt_every,
        sigma=args.sigma, noise_seed=args.noise_seed)
