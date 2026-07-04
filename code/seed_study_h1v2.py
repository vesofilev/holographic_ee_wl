# ASSERT_CONVENTION: AdS5-Schwarzschild h=1, f(z)=1-z^4, z_h=1, d=4
# ASSERT_CONVENTION: same S_EE data as the paper's Sec 5.1 (50 pts, l in [0.15, 0.75 l_c])
"""
Upgraded h=1 inverse (AdS5-Schwarzschild validation, paper Sec 5.1).

The original protocol (ratio-normalization V-model encoding + data-loss
weight 100) exhibits a long, seed-dependent initialization plateau and lands
at ~1.7% max relative error. This runner keeps the DATA identical to the
paper (50 points on l in [0.15, 0.75 l_c]) and upgrades the optimization:

  --encoding structured : f = 1 + a*z + z^2[-(1+a) + D(z) - D(1)]
                          (GR-style ansatz: f(0)=1, f(1)=0, trainable a,
                          well-conditioned gradients, no ratio)
  --encoding ratio      : original VModel (for baseline comparison)
  --weight W            : data-loss weight (paper: 100; GR study: 10000)

Everything else matches the paper protocol: LModel(16,2), lr_L=1e-4,
lr_V=5e-4, n_grid=4000, alternating V/L steps.

Usage:  python seed_study_h1v2.py --seed 1 --encoding structured --weight 10000
Output: data/seed_study/h1v2/h1v2_<enc>_w<W>_s<seed>.npz (+ _models.pt)
"""
import os
import sys
import time
import math
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('seed_study_h1v2')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d

torch.set_num_threads(1)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from utils import make_half_grid, load_ode_benchmark, Z_H
from ann_forward import X_S_FRAC
from ann_inverse import LModel, VModel, rt_area_V

OUT_DIR = os.path.join(BASE_DIR, 'data', 'seed_study', 'h1v2')
ERR_STRIDE = 5000


class VModelH1Structured(nn.Module):
    """GR-style structured encoding for a single blackening factor:
        f(z) = 1 + a*zn + zn^2 * [-(1 + a) + D(zn) - D(1)],   zn = z/z_h,
    so that f(0) = 1 and f(z_h) = 0 exactly, with trainable boundary
    slope a (the exact AdS-Schwarzschild solution has a = 0, f = 1 - zn^4)."""

    def __init__(self, z_h=Z_H, hidden=20, depth=2, fix_a=False):
        super().__init__()
        self.z_h = z_h
        if fix_a:
            # physics-fixed boundary slope: undeformed CFT has no modes
            # below the normalizable order, so f'(0) = 0 identically
            self.a = torch.tensor(0.0, dtype=torch.float64)
        else:
            self.a = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers).double()
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.05)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        if z.dim() == 0:
            z = z.unsqueeze(0)
        orig_shape = z.shape
        zn = (z / self.z_h).reshape(-1, 1)
        D = self.net(zn) - self.net(torch.ones(1, 1, dtype=z.dtype))
        f = 1.0 + self.a * zn + zn ** 2 * (-(1.0 + self.a) + D)
        return f.squeeze(-1).reshape(orig_shape)


def f_metrics(v_model, z_eval):
    z_t = torch.tensor(z_eval, dtype=torch.float64)
    with torch.no_grad():
        f_pred = v_model(z_t).numpy()
    f_ex = 1.0 - z_eval ** 4
    abs_err = np.abs(f_pred - f_ex)
    m = z_eval < 0.95
    rel = abs_err[m] / np.abs(f_ex[m])
    return f_pred, abs_err.max(), abs_err.mean(), rel.max(), rel.mean()


def run(seed, encoding, weight, n_epochs, lr_L=1e-4, lr_V=5e-4, n_grid=4000,
        resume_from=None, ckpt_every=50000):
    ode = load_ode_benchmark(os.path.join(BASE_DIR, 'data', 'ode_benchmark.npz'))
    l_c = ode['l_c']
    l_to_zs = interp1d(ode['l_of_zstar'], ode['z_star_grid'],
                       kind='cubic', fill_value='extrapolate')
    zs_to_Areg = interp1d(ode['z_star_grid'], ode['A_reg_of_zstar'],
                          kind='cubic', fill_value='extrapolate')
    l_data = np.linspace(0.15, 0.75 * l_c, 50)          # identical to paper
    A_data = np.array([float(zs_to_Areg(float(l_to_zs(lv)))) for lv in l_data])
    l_min, l_max = float(l_data.min()), float(l_data.max())
    S_interp = interp1d(l_data, A_data, kind='cubic', fill_value='extrapolate')

    if resume_from is None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    l_model = LModel(hidden=16, depth=2)
    if encoding in ('structured', 'structured0'):
        v_model = VModelH1Structured(z_h=Z_H, hidden=20, depth=2,
                                     fix_a=(encoding == 'structured0'))
    else:
        v_model = VModel(z_h=Z_H, hidden=16, depth=2)
    opt_L = optim.Adam(l_model.parameters(), lr=lr_L)
    opt_V = optim.Adam(v_model.parameters(), lr=lr_V)

    start_epoch = 1
    if resume_from is not None:
        ck = torch.load(resume_from, weights_only=False)
        l_model.load_state_dict(ck['l_model'])
        v_model.load_state_dict(ck['v_model'])
        if ck.get('opt_L') is not None:
            opt_L.load_state_dict(ck['opt_L'])
            opt_V.load_state_dict(ck['opt_V'])
        if ck.get('torch_rng') is not None:
            torch.set_rng_state(ck['torch_rng'])
            np.random.set_state(ck['np_rng'])
        start_epoch = int(ck['epoch']) + 1
        logger.info(f"  Resumed from {resume_from} (epoch {start_epoch - 1})")

    base = f"h1v2_{encoding}_w{int(weight)}_s{seed:02d}"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_npz = os.path.join(OUT_DIR, base + '.npz')

    def _save_ckpt(ep):
        torch.save({'l_model': l_model.state_dict(),
                    'v_model': v_model.state_dict(),
                    'opt_L': opt_L.state_dict(), 'opt_V': opt_V.state_dict(),
                    'torch_rng': torch.get_rng_state(),
                    'np_rng': np.random.get_state(),
                    'seed': seed, 'encoding': encoding, 'weight': weight,
                    'epoch': ep},
                   out_npz.replace('.npz', '_models.pt'))

    logger.info(f"seed={seed} encoding={encoding} weight={weight} "
                f"epochs={start_epoch}..{n_epochs}  data: {len(l_data)} pts "
                f"l=[{l_min:.4f},{l_max:.4f}]")

    z_eval = np.linspace(0, Z_H, 200)
    err_ep, abs_hist, rel_hist, a_hist = [], [], [], []

    t0 = time.time()
    for epoch in range(start_epoch, n_epochs + 1):
        l_val = float(np.random.uniform(l_min, l_max))
        S_target = float(S_interp(l_val))
        x_np = make_half_grid(l_val, n_grid, x_s_frac=X_S_FRAC)
        A_reg = rt_area_V(l_model, v_model, x_np, l_val)

        if epoch % 2 == 0:
            opt_V.zero_grad()
            loss = weight * (A_reg - S_target) ** 2
            loss.backward()
            opt_V.step()
        else:
            opt_L.zero_grad()
            loss = A_reg
            loss.backward()
            opt_L.step()

        if epoch % ERR_STRIDE == 0:
            _, mxa, mna, mxr, mnr = f_metrics(v_model, z_eval)
            err_ep.append(epoch)
            abs_hist.append(mxa)
            rel_hist.append(mxr)
            a_hist.append(v_model.a.item() if encoding == 'structured' else np.nan)

        if epoch % 25000 == 0 or epoch == start_epoch:
            el = time.time() - t0
            rate = (epoch - start_epoch + 1) / el if el > 0 else 0
            mxa = abs_hist[-1] if abs_hist else float('nan')
            mxr = rel_hist[-1] if rel_hist else float('nan')
            logger.info(f"  ep {epoch:7d}/{n_epochs}  max|df|={mxa:.5f}  "
                        f"max|df/f|={mxr:.4f}  [{el/60:.1f}min {rate:.0f}ep/s]")

        if epoch % ckpt_every == 0:
            _save_ckpt(epoch)

    wall_min = (time.time() - t0) / 60.0
    f_final, max_abs, mean_abs, max_rel, mean_rel = f_metrics(v_model, z_eval)

    # learned S_EE(l) residuals (80 pts, as in the paper's evaluation)
    l_model.eval(); v_model.eval()
    l_eval = np.linspace(l_min, l_max, 80)
    A_ann = []
    for lv in l_eval:
        x_np = make_half_grid(lv, 4000, x_s_frac=X_S_FRAC)
        A_ann.append(rt_area_V(l_model, v_model, x_np, lv).item())
    A_ann = np.array(A_ann)
    A_ref = np.array([float(S_interp(lv)) for lv in l_eval])

    np.savez(out_npz,
             seed=seed, encoding=encoding, weight=weight,
             n_epochs=n_epochs, wall_min=wall_min,
             l_data=l_data, A_data=A_data,
             z_eval=z_eval, f_final=f_final, f_exact=1.0 - z_eval ** 4,
             max_abs_err=max_abs, mean_abs_err=mean_abs,
             max_rel_err=max_rel, mean_rel_err=mean_rel,
             err_ep=np.array(err_ep), abs_hist=np.array(abs_hist),
             rel_hist=np.array(rel_hist), a_hist=np.array(a_hist),
             l_eval=l_eval, A_ann=A_ann, A_ref=A_ref)
    _save_ckpt(n_epochs)

    logger.info(f"DONE {base}: max|df|={max_abs:.5f} mean|df|={mean_abs:.5f}  "
                f"max|df/f|={max_rel:.4f} mean={mean_rel:.4f} (z<0.95)  "
                f"S-fit RMS={np.sqrt(np.mean((A_ann-A_ref)**2)):.5f}  "
                f"[{wall_min:.1f} min]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--encoding', choices=['structured', 'structured0', 'ratio'], default='structured0')
    ap.add_argument('--weight', type=float, default=10000.0)
    ap.add_argument('--epochs', type=int, default=500000)
    ap.add_argument('--resume-from', type=str, default=None)
    ap.add_argument('--ckpt-every', type=int, default=50000)
    args = ap.parse_args()
    run(args.seed, args.encoding, args.weight, args.epochs,
        resume_from=args.resume_from, ckpt_every=args.ckpt_every)
