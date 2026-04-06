# ASSERT_CONVENTION: AdS4 (d=3 boundary) Gubser-Rocha
# ASSERT_CONVENTION: ds^2 = (1/z^2)[-f dt^2 + dz^2/f + h(dx^2+dy^2)]
# ASSERT_CONVENTION: RT area = Omega int (sqrt(h)/z^2) sqrt(h + z'^2/f) dx
# ASSERT_CONVENTION: WL potential = (1/2pi alpha') int (1/z^2) sqrt(f*h + z'^2) dx
# ASSERT_CONVENTION: inverse: learn f(z) AND h(z) from S_EE(l) + V(L) + thermal entropy
"""
Inverse problem for AdS4 Gubser-Rocha with combined S_EE + Wilson loop data.

Three-network approach:
  L-model: RT surface z_L(x,l) — minimizes area functional
  W-model: Wilson loop string z_W(x,L) — minimizes Nambu-Goto action
  V-model: metric f(z), h(z) — shared between L and W, trained on combined loss

This resolves the metric degeneracy that afflicts S_EE-only reconstruction.
"""
import os
import sys
import time
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import interp1d

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("inverse_gr_d3_wl_training.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import make_half_grid, Z_H
from ann_inverse import LModel
from ann_forward import EPSILON, X_S_FRAC
from ann_inverse_gr_d3 import (VModelGR_d3, ExactVModel_d3,
                                rt_area_V_gr_d3, rt_area_V_gr_d3_batch,
                                _save_snapshot_d3, D_BOUNDARY_D3)

rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# =============================================================================
# W-model: Wilson loop string profile z_W(x, L)
# Same architecture as L-model but separate parameters
# =============================================================================
class WModel(nn.Module):
    """Wilson loop string profile z_W(x, L).
    Same BC encoding as LModel: z = eps + (L^2/4 - x^2)^{1/d} * softplus(g(x^2,L)).
    For d=3, the string profile has the same near-boundary asymptotics as the RT surface.
    """
    def __init__(self, hidden=16, depth=2, d=D_BOUNDARY_D3, epsilon=EPSILON):
        super().__init__()
        self.d = d
        self.epsilon = epsilon
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.core = nn.Sequential(*layers).double()
        self._init_weights()

    def _init_weights(self):
        for m in self.core:
            if isinstance(m, nn.Linear):
                fan = m.weight.size(0) + m.weight.size(1)
                nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / fan) * 0.5)
                nn.init.zeros_(m.bias)
        final = self.core[-1]
        nn.init.normal_(final.weight, 0.0, 0.01)
        nn.init.zeros_(final.bias)

    def forward(self, x, L_val):
        if not isinstance(L_val, torch.Tensor):
            L_t = torch.full_like(x, L_val)
        else:
            L_t = L_val * torch.ones_like(x)
        x_sq = (x / (L_t / 2.0)) ** 2
        inp = torch.cat([x_sq, L_t], dim=1)
        g = self.core(inp)
        prefactor = (L_t / 2.0) ** 2 - x ** 2
        prefactor = torch.clamp(prefactor, min=0.0)
        return self.epsilon + prefactor ** (1.0 / self.d) * nn.functional.softplus(g)


# =============================================================================
# Wilson loop action (Nambu-Goto) for AdS4 GR
# =============================================================================
def wl_action_V_gr_d3_batch(w_model, v_model, L_vals, x_s_frac=X_S_FRAC,
                             epsilon=EPSILON, z_h=Z_H, n_grid=4000,
                             create_graph=True):
    """
    Batched V_reg for Wilson loop.

    V(L) = (1/2pi alpha') int (1/z^2) sqrt(f*h + z'^2) dx
    V_reg = V_conn - 2 * int_eps^{z_h} (1/z^2) dz  (straight strings)

    We use the same hybrid x/z split as the RT area:
      Interior (x < x_s): integrate in x-parametrization
      Boundary (x > x_s): switch to z-parametrization, subtract disconnected

    For Wilson loop:
      Connected integrand (x-param): (1/z^2) sqrt(f*h + z'^2)
      Connected integrand (z-param): (1/z^2) sqrt(f*h*x'^2 + 1)  where x'=1/z'
      Disconnected integrand: 1/z^2  (straight string: sqrt(f/z^2 * 1/(z^2*f)) = 1/z^2)
      Subtracted: (1/z^2)(sqrt(f*h*x'^2 + 1) - 1)
      Remainder:  int_{z_mid}^{z_h} 1/z^2 dz = 1/z_mid - 1/z_h
    """
    B = len(L_vals)
    n_int = max(int(n_grid * 0.3), 100)
    n_bdy = n_grid - n_int
    u_s = 2.0 * x_s_frac

    u_int = np.linspace(0, u_s, n_int, endpoint=False)
    u_bdy = np.linspace(u_s, 1.0 - 1e-8, n_bdy)
    u_grid = np.concatenate([u_int, u_bdy])
    u_t = torch.tensor(u_grid, dtype=torch.float64)

    L_arr = torch.tensor(L_vals, dtype=torch.float64)
    x_all = u_t.unsqueeze(0) * (L_arr.unsqueeze(1) / 2.0)

    x_flat = x_all.reshape(-1, 1).clone().requires_grad_(True)
    L_flat = L_arr.unsqueeze(1).expand(B, len(u_grid)).reshape(-1, 1)

    z_flat = w_model(x_flat, L_flat)
    dzdx_flat = torch.autograd.grad(z_flat.sum(), x_flat, create_graph=create_graph)[0]

    z_all = z_flat.squeeze(1).reshape(B, -1)
    dzdx_all = dzdx_flat.squeeze(1).reshape(B, -1)

    f_all = torch.clamp(v_model.forward_f(z_all.reshape(-1)).reshape(B, -1), min=1e-12)
    h_all = torch.clamp(v_model.forward_h(z_all.reshape(-1)).reshape(B, -1), min=1e-12)

    mask_int = torch.tensor(u_grid <= u_s + 1e-14)
    mask_bdy = ~mask_int

    # --- Interior: (1/z^2) sqrt(f*h + z'^2) ---
    z_int = z_all[:, mask_int]
    dz_int = dzdx_all[:, mask_int]
    f_int = f_all[:, mask_int]
    h_int = h_all[:, mask_int]
    integ_int = (1.0 / z_int**2) * torch.sqrt(f_int * h_int + dz_int**2)
    x_int = x_all[:, mask_int]
    I_interior = torch.trapezoid(integ_int, x_int, dim=1)

    # --- Boundary: (1/z^2)(sqrt(f*h*x'^2 + 1) - 1) in z-parametrization ---
    z_bdy = z_all[:, mask_bdy]
    dz_bdy = dzdx_all[:, mask_bdy]
    f_bdy_v = f_all[:, mask_bdy]
    h_bdy_v = h_all[:, mask_bdy]
    dzdx_safe = torch.clamp(dz_bdy, max=-1e-10)
    xp_bdy = 1.0 / dzdx_safe

    conn_z = torch.sqrt(f_bdy_v * h_bdy_v * xp_bdy**2 + 1.0)
    disc_z = 1.0  # straight string: 1/z^2 * dz, so just 1 after factoring 1/z^2
    g_z = (1.0 / z_bdy**2) * (conn_z - disc_z)

    z_sorted, idx = torch.sort(z_bdy, dim=1)
    g_sorted = torch.gather(g_z, 1, idx)
    I_bdy = torch.trapezoid(g_sorted, z_sorted, dim=1)

    # --- Remainder: int_{z_mid}^{z_h} 1/z^2 dz = 1/z_mid - 1/z_h ---
    z_sorted_for_mid, _ = torch.sort(z_bdy, dim=1)
    z_mid = z_sorted_for_mid[:, -1]
    I_rem = 1.0 / z_mid - 1.0 / z_h

    return list(I_interior + I_bdy - I_rem)


def wl_action_V_gr_d3(w_model, v_model, x_grid_np, L_val, **kwargs):
    """Single-L Wilson loop action."""
    return wl_action_V_gr_d3_batch(w_model, v_model, [L_val],
                                    n_grid=len(x_grid_np), **kwargs)[0]


# =============================================================================
# Snapshot plot
# =============================================================================
def _save_snapshot_wl(v_model, l_model, w_model,
                      S_interp, V_interp,
                      l_range, L_range, epoch, Q, plot_dir):
    """6-panel snapshot: f(z), h(z), S_EE(l), V(L), f error, h error."""
    from ode_benchmark_gr_d3 import f_exact, h_exact

    z_eval = np.linspace(0.01, 0.99, 200)
    z_t = torch.tensor(z_eval, dtype=torch.float64)
    with torch.no_grad():
        f_pred = v_model.forward_f(z_t).numpy()
        h_pred = v_model.forward_h(z_t).numpy()
    f_ex = np.array([f_exact(z, Q) for z in z_eval])
    h_ex = np.array([h_exact(z, Q) for z in z_eval])

    # S_EE comparison
    l_min, l_max = l_range
    l_arr = np.linspace(l_min, l_max, 20)
    A_pred, A_ref = [], []
    l_model.eval(); v_model.eval()
    for lv in l_arr:
        x_np = make_half_grid(lv, 2000, x_s_frac=X_S_FRAC)
        A_pred.append(rt_area_V_gr_d3(l_model, v_model, x_np, lv).item())
        A_ref.append(float(S_interp(lv)))

    # V(L) comparison
    L_min, L_max = L_range
    L_arr = np.linspace(L_min, L_max, 20)
    V_pred, V_ref = [], []
    w_model.eval()
    for Lv in L_arr:
        x_np = make_half_grid(Lv, 2000, x_s_frac=X_S_FRAC)
        V_pred.append(wl_action_V_gr_d3(w_model, v_model, x_np, Lv).item())
        V_ref.append(float(V_interp(Lv)))
    l_model.train(); w_model.train(); v_model.train()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(z_eval, f_ex, 'r-', lw=1.5, label='Exact')
    axes[0, 0].plot(z_eval, f_pred, 'b--', lw=1.5, label='Learned')
    axes[0, 0].set_xlabel(r'$z$'); axes[0, 0].set_ylabel(r'$f(z)$')
    axes[0, 0].set_title(f'f(z) (ep {epoch})')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(z_eval, h_ex, 'r-', lw=1.5, label='Exact')
    axes[0, 1].plot(z_eval, h_pred, 'b--', lw=1.5, label='Learned')
    axes[0, 1].set_xlabel(r'$z$'); axes[0, 1].set_ylabel(r'$h(z)$')
    axes[0, 1].set_title(f'h(z) (ep {epoch})')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(l_arr, A_ref, 'r.-', lw=1, label='Data')
    axes[0, 2].plot(l_arr, A_pred, 'b.--', lw=1, label='Learned')
    axes[0, 2].set_xlabel(r'$l$'); axes[0, 2].set_ylabel(r'$S_{EE}$')
    axes[0, 2].set_title(r'$S_{EE}(l)$')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(L_arr, V_ref, 'r.-', lw=1, label='Data')
    axes[1, 0].plot(L_arr, V_pred, 'b.--', lw=1, label='Learned')
    axes[1, 0].set_xlabel(r'$L$'); axes[1, 0].set_ylabel(r'$V(L)$')
    axes[1, 0].set_title(r'$V(L)$ Wilson loop')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    mask = f_ex > 0.01
    axes[1, 1].semilogy(z_eval[mask], np.abs(f_pred[mask] - f_ex[mask]) / f_ex[mask],
                         'b-', lw=1, label=r'$|\Delta f/f|$')
    axes[1, 1].semilogy(z_eval, np.abs(h_pred - h_ex) / h_ex,
                         'r-', lw=1, label=r'$|\Delta h/h|$')
    axes[1, 1].set_xlabel(r'$z$'); axes[1, 1].set_ylabel('Relative error')
    axes[1, 1].set_title('Metric errors')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_ylim(1e-5, 10)

    a_val = v_model.a.item()
    axes[1, 2].text(0.1, 0.7, f'epoch = {epoch}', transform=axes[1, 2].transAxes, fontsize=14)
    axes[1, 2].text(0.1, 0.55, f'a = {a_val:.4f} (exact {1.5*Q:.4f})', transform=axes[1, 2].transAxes, fontsize=12)
    with torch.no_grad():
        h_zh = v_model.forward_h(torch.tensor([Z_H], dtype=torch.float64)).item()
    axes[1, 2].text(0.1, 0.4, f'h(z_h) = {h_zh:.4f} (exact {(1+Q)**1.5:.4f})', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.25, f'max |f err| = {np.max(np.abs(f_pred[mask]-f_ex[mask])/f_ex[mask]):.4f}', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].text(0.1, 0.1, f'max |h err| = {np.max(np.abs(h_pred-h_ex)/h_ex):.4f}', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'snapshot_live.png'), dpi=150)
    plt.close()


# =============================================================================
# Training with three-way alternating optimization
# =============================================================================
def train_inverse_gr_d3_wl(S_data_l, S_data_A, V_data_L, V_data_V,
                            Q, n_epochs=500000,
                            lr_L=1e-4, lr_W=1e-4, lr_V=5e-4,
                            n_grid=4000, print_every=2000,
                            checkpoint_every=50000, seed=42,
                            lambda_s=1.0, resume_from=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(base_dir, 'data', 'inverse_gr_d3_wl', 'checkpoints')
    plot_dir = os.path.join(base_dir, 'plots', 'inverse_gr_d3_wl')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    l_min, l_max = float(S_data_l.min()), float(S_data_l.max())
    L_min, L_max = float(V_data_L.min()), float(V_data_L.max())
    S_interp = interp1d(S_data_l, S_data_A, kind='cubic', fill_value='extrapolate')
    V_interp = interp1d(V_data_L, V_data_V, kind='cubic', fill_value='extrapolate')

    s_target = (1.0 + Q) ** 1.5

    l_model = LModel(hidden=32, depth=2, d=D_BOUNDARY_D3)
    w_model = WModel(hidden=32, depth=2, d=D_BOUNDARY_D3)
    v_model = VModelGR_d3(z_h=Z_H, h_h=s_target, hidden=20, depth=2)

    start_epoch = 1
    if resume_from is not None:
        ckpt = torch.load(resume_from, weights_only=False)
        l_model.load_state_dict(ckpt['l_model'])
        w_model.load_state_dict(ckpt['w_model'])
        v_model.load_state_dict(ckpt['v_model'])
        start_epoch = ckpt['epoch'] + 1
        logger.info(f"  Resumed from {resume_from} (epoch {ckpt['epoch']})")

    opt_L = optim.Adam(l_model.parameters(), lr=lr_L)
    opt_W = optim.Adam(w_model.parameters(), lr=lr_W)
    opt_V = optim.Adam(v_model.parameters(), lr=lr_V)

    n_L = sum(p.numel() for p in l_model.parameters())
    n_W = sum(p.numel() for p in w_model.parameters())
    n_V = sum(p.numel() for p in v_model.parameters())
    logger.info(f"  L: {n_L}, W: {n_W}, V: {n_V} params")
    logger.info(f"  lr_L={lr_L}, lr_W={lr_W}, lr_V={lr_V}, epochs={n_epochs}")
    logger.info(f"  Q={Q}, s_target={s_target:.6f}, lambda_s={lambda_s}")
    logger.info(f"  S_EE range: l=[{l_min:.4f}, {l_max:.4f}]")
    logger.info(f"  V(L) range: L=[{L_min:.4f}, {L_max:.4f}]")

    from ode_benchmark_gr_d3 import f_exact, h_exact
    loss_hist = []
    t0 = time.time()

    for epoch in range(start_epoch, n_epochs + 1):
        step_type = epoch % 4  # 0: V-step, 1: L-step, 2: V-step, 3: W-step

        if step_type in (0, 2):
            # V-step: update metric on combined data loss
            opt_V.zero_grad()

            # S_EE loss
            l_val = float(np.random.uniform(l_min, l_max))
            S_target = float(S_interp(l_val))
            x_np = make_half_grid(l_val, n_grid, x_s_frac=X_S_FRAC)
            A_reg = rt_area_V_gr_d3(l_model, v_model, x_np, l_val)
            see_loss = 10000.0 * (A_reg - S_target)**2

            # Wilson loop loss
            L_val = float(np.random.uniform(L_min, L_max))
            V_target = float(V_interp(L_val))
            V_reg = wl_action_V_gr_d3(w_model, v_model, x_np, L_val)
            wl_loss = 10000.0 * (V_reg - V_target)**2

            loss = see_loss + wl_loss
            loss.backward()
            opt_V.step()

        elif step_type == 1:
            # L-step: minimize RT area
            l_val = float(np.random.uniform(l_min, l_max))
            x_np = make_half_grid(l_val, n_grid, x_s_frac=X_S_FRAC)
            A_reg = rt_area_V_gr_d3(l_model, v_model, x_np, l_val)
            opt_L.zero_grad()
            loss = A_reg
            loss.backward()
            opt_L.step()

        elif step_type == 3:
            # W-step: minimize Wilson loop action
            L_val = float(np.random.uniform(L_min, L_max))
            x_np = make_half_grid(L_val, n_grid, x_s_frac=X_S_FRAC)
            V_reg = wl_action_V_gr_d3(w_model, v_model, x_np, L_val)
            opt_W.zero_grad()
            loss = V_reg
            loss.backward()
            opt_W.step()

        loss_hist.append(loss.item())

        if epoch % print_every == 0 or epoch == 1:
            z_pts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            z_test = torch.tensor(z_pts, dtype=torch.float64)
            with torch.no_grad():
                f_test = v_model.forward_f(z_test).numpy()
                h_test = v_model.forward_h(z_test).numpy()
            f_ex = np.array([f_exact(z, Q) for z in z_pts])
            h_ex = np.array([h_exact(z, Q) for z in z_pts])
            f_err = np.max(np.abs(f_test - f_ex) / np.maximum(np.abs(f_ex), 1e-10))
            h_err = np.max(np.abs(h_test - h_ex) / h_ex)

            elapsed = time.time() - t0
            rate = epoch / elapsed if elapsed > 0 else 0
            a_val = v_model.a.item()
            logger.info(
                f"  ep {epoch:7d}/{n_epochs}  "
                f"a={a_val:.4f}({1.5*Q:.4f})  "
                f"|f err|={f_err:.4f}  |h err|={h_err:.4f}  "
                f"[{elapsed/60:.1f}min {rate:.0f}ep/s]"
            )
            f_line = "    f: " + ", ".join(f"f({z})={v:.3f}({e:.3f})" for z, v, e in zip(z_pts, f_test, f_ex))
            h_line = "    h: " + ", ".join(f"h({z})={v:.3f}({e:.3f})" for z, v, e in zip(z_pts, h_test, h_ex))
            logger.info(f_line)
            logger.info(h_line)

        if epoch % print_every == 0:
            _save_snapshot_wl(v_model, l_model, w_model,
                              S_interp, V_interp,
                              (l_min, l_max), (L_min, L_max),
                              epoch, Q, plot_dir)

        if epoch % checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'l_model': l_model.state_dict(),
                'w_model': w_model.state_dict(),
                'v_model': v_model.state_dict(),
            }, os.path.join(ckpt_dir, f"ep{epoch:06d}.pt"))
            logger.info(f"  >> Checkpoint saved: ep {epoch}")

    t_train = time.time() - t0
    logger.info(f"  Training done: {t_train/60:.1f} min")
    return l_model, w_model, v_model, loss_hist


# =============================================================================
# Main: generate data + train
# =============================================================================
if __name__ == "__main__":
    Q = 1.0

    # Load ODE benchmark data for S_EE
    data_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'data', 'ode_benchmark_gr_d3.npz')
    if not os.path.exists(data_path):
        logger.info("Generating ODE benchmark data...")
        import ode_benchmark_gr_d3
        ode_benchmark_gr_d3.main()

    data = np.load(data_path)
    l_data = data['l_of_zstar']
    A_data = data['A_reg_of_zstar']
    mask = (l_data > 0.15) & (A_data < 0)  # connected branch
    S_data_l = l_data[mask]
    S_data_A = A_data[mask]

    # Generate Wilson loop data
    logger.info("Generating Wilson loop benchmark data...")
    from bilson_reconstruction_v2 import (compute_L_WL, compute_Vreg_WL,
                                           f_exact as f_ex_func,
                                           h_exact as h_ex_func)

    z_stars = np.linspace(0.15, 0.90, 150)
    V_data_L_arr = np.array([compute_L_WL(zs) for zs in z_stars])
    V_data_V_arr = np.array([compute_Vreg_WL(zs) for zs in z_stars])
    logger.info(f"  Wilson loop data: {len(V_data_L_arr)} points, "
                f"L=[{V_data_L_arr.min():.4f}, {V_data_L_arr.max():.4f}]")

    # Train
    n_epochs = 500000
    logger.info(f"\nStarting 3-network training ({n_epochs} epochs)...")
    l_model, w_model, v_model, loss_hist = train_inverse_gr_d3_wl(
        S_data_l, S_data_A,
        V_data_L_arr, V_data_V_arr,
        Q=Q, n_epochs=n_epochs,
        lr_L=1e-4, lr_W=1e-4, lr_V=5e-4,
        n_grid=4000, lambda_s=10000.0,
        print_every=5000, checkpoint_every=50000
    )
    logger.info("DONE.")
