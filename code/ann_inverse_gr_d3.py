# ASSERT_CONVENTION: AdS4 (d=3 boundary) Gubser-Rocha
# ASSERT_CONVENTION: ds^2 = (1/z^2)[-f dt^2 + dz^2/f + h(dx^2+dy^2)]
# ASSERT_CONVENTION: area = Omega int (sqrt(h)/z^2) sqrt(h + z'^2/f) dx
# ASSERT_CONVENTION: inverse: learn f(z) AND h(z) from S_EE(l) + thermal entropy
"""
Inverse problem for AdS4 Gubser-Rocha: learn f(z) and h(z) from S_EE(l).

Uses the same variational ANN approach as the AdS5 code, adapted for d=3:
  - Area integrand: (sqrt(h)/z^2) sqrt(h + z'^2/f)
  - Disconnected: sqrt(h)/(z^2 sqrt(f))
  - h encoding: h = 1 + z*g(z) (h(0)=1, h(z_h) free, h'(0) free)
  - Loss: data loss + thermal entropy penalty

References:
  - Ahn et al., arXiv:2406.07395
  - Li and Liu, arXiv:2307.04433
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
        logging.FileHandler("inverse_gr_d3_training.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import make_half_grid, Z_H
from ann_inverse import LModel
from ann_forward import EPSILON, X_S_FRAC

# Override D_BOUNDARY for d=3
D_BOUNDARY_D3 = 3

rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# =============================================================================
# VModelGR_d3: Two networks for f(z) and h(z) [AdS4]
# =============================================================================
class VModelGR_d3(nn.Module):
    """
    Ahn et al. ansatz:
      f(z) = (1 - z/z_h)(1 + a*z/z_h + (z/z_h)^2 * D_f(z/z_h))
      h(z) = 1 + a*z/z_h + (z/z_h)^2 * D_h(z/z_h)
    where a is a shared trainable scalar, D_f and D_h are networks.
    f(0)=1, f(z_h)=0, h(0)=1 by construction.
    f'(0) = h'(0) = a/z_h structurally (UV regularity).
    h(z_h) is free — enforced by thermal entropy penalty in the loss.
    """
    def __init__(self, z_h=Z_H, h_h=None, hidden=20, depth=2):
        super().__init__()
        self.z_h = z_h
        self.h_h = h_h  # only used for thermal entropy penalty

        # Shared boundary derivative parameter
        self.a = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

        # D_f network
        f_layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            f_layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        f_layers += [nn.Linear(hidden, 1)]
        self.f_net = nn.Sequential(*f_layers).double()

        # D_h network
        h_layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            h_layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        h_layers += [nn.Linear(hidden, 1)]
        self.h_net = nn.Sequential(*h_layers).double()

        self._init_weights()

    def _init_weights(self):
        for net in [self.f_net, self.h_net]:
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, 0.05)
                    nn.init.zeros_(m.bias)
            net[-1].bias.data.fill_(0.0)  # D starts at ~0

    def forward_f(self, z):
        """f(z) = 1 + a*z/z_h + (z/z_h)^2 * D_f(z/z_h),
        with D_f(1) = -(1+a) so that f(1) = 0.
        D_f(z) = -(1+a) + (net(z) - net(1))."""
        if z.dim() == 0:
            z = z.unsqueeze(0)
        orig_shape = z.shape
        zn = (z / self.z_h).reshape(-1, 1)
        net_z = self.f_net(zn)
        net_1 = self.f_net(torch.ones(1, 1, dtype=z.dtype, device=z.device))
        D = -(1.0 + self.a) + (net_z - net_1)
        f = 1.0 + self.a * zn + zn**2 * D
        return f.squeeze(-1).reshape(orig_shape)

    def forward_h(self, z):
        """h(z) = 1 + a*z/z_h + (z/z_h)^2 * D_h(z/z_h)."""
        if z.dim() == 0:
            z = z.unsqueeze(0)
        orig_shape = z.shape
        zn = (z / self.z_h).reshape(-1, 1)
        D = self.h_net(zn)
        h = 1.0 + self.a * zn + zn**2 * D
        return h.squeeze(-1).reshape(orig_shape)

    def forward(self, z):
        return self.forward_f(z), self.forward_h(z)


class ExactVModel_d3(torch.nn.Module):
    """Exact GR metric for on-the-fly target computation (no trainable params).
    Returns f_exact, h_exact as non-differentiable tensors."""
    def __init__(self, Q):
        super().__init__()
        self.Q = Q

    def forward_f(self, z):
        z_np = z.detach().cpu().numpy()
        Q = self.Q
        U = (1 + (1+3*Q)*z_np + (1+3*Q+3*Q**2)*z_np**2) / (1+Q*z_np)**1.5
        f_np = (1 - z_np) * U
        return torch.tensor(f_np, dtype=z.dtype, device=z.device)

    def forward_h(self, z):
        z_np = z.detach().cpu().numpy()
        return torch.tensor((1 + self.Q * z_np) ** 1.5,
                            dtype=z.dtype, device=z.device)


# =============================================================================
# Area functional for AdS4 GR (d=3)
# =============================================================================
def rt_area_V_gr_d3(l_model, v_model, x_grid_np, l_val,
                    x_s_frac=X_S_FRAC, epsilon=EPSILON, z_h=Z_H,
                    use_mc=False):
    """Single-l area computation (used for L-steps and diagnostics)."""
    return rt_area_V_gr_d3_batch(l_model, v_model, [l_val],
                                  x_s_frac=x_s_frac, epsilon=epsilon,
                                  z_h=z_h, n_grid=len(x_grid_np),
                                  use_mc=use_mc)[0]


def rt_area_V_gr_d3_batch(l_model, v_model, l_vals, x_s_frac=X_S_FRAC,
                           epsilon=EPSILON, z_h=Z_H, n_grid=4000,
                           create_graph=True, use_mc=False):
    """
    Batched A_reg_half for multiple l values.
    Uses a shared normalized grid u in [0,1], scaled by l/2 for each l.
    use_mc=False: trapezoid on fixed grid (original).
    use_mc=True:  Monte Carlo on random samples.
    Returns list of A_reg tensors (one per l, all in the same graph).
    """
    B = len(l_vals)
    n_int = max(int(n_grid * 0.3), 100)
    n_bdy = n_grid - n_int
    u_s = 2.0 * x_s_frac  # normalized split = 0.4

    if use_mc:
        u_int = np.random.uniform(0, u_s, n_int)
        u_bdy = np.random.uniform(u_s, 1.0 - 1e-8, n_bdy)
    else:
        u_int = np.linspace(0, u_s, n_int, endpoint=False)
        u_bdy = np.linspace(u_s, 1.0 - 1e-8, n_bdy)
    u_grid = np.concatenate([u_int, u_bdy])
    u_t = torch.tensor(u_grid, dtype=torch.float64)

    l_arr = torch.tensor(l_vals, dtype=torch.float64)

    # x = u * l/2
    x_all = u_t.unsqueeze(0) * (l_arr.unsqueeze(1) / 2.0)  # (B, G)

    x_flat = x_all.reshape(-1, 1).clone().requires_grad_(True)
    l_flat = l_arr.unsqueeze(1).expand(B, len(u_grid)).reshape(-1, 1)

    z_flat = l_model(x_flat, l_flat)
    dzdx_flat = torch.autograd.grad(z_flat.sum(), x_flat, create_graph=create_graph)[0]

    z_all = z_flat.squeeze(1).reshape(B, -1)
    dzdx_all = dzdx_flat.squeeze(1).reshape(B, -1)

    f_all = torch.clamp(v_model.forward_f(z_all.reshape(-1)).reshape(B, -1), min=1e-12)
    h_all = torch.clamp(v_model.forward_h(z_all.reshape(-1)).reshape(B, -1), min=1e-12)

    mask_int = torch.tensor(u_grid <= u_s + 1e-14)
    mask_bdy = ~mask_int

    # --- Interior ---
    z_int = z_all[:, mask_int]
    dz_int = dzdx_all[:, mask_int]
    f_int = f_all[:, mask_int]
    h_int = h_all[:, mask_int]
    integ_int = (torch.sqrt(h_int) / z_int**2) * torch.sqrt(
        h_int + dz_int**2 / f_int)
    if use_mc:
        x_s_val = l_arr / 2.0 * u_s
        I_interior = x_s_val * integ_int.mean(dim=1)
    else:
        x_int = x_all[:, mask_int]
        I_interior = torch.trapezoid(integ_int, x_int, dim=1)

    # --- Boundary ---
    z_bdy = z_all[:, mask_bdy]
    dz_bdy = dzdx_all[:, mask_bdy]
    f_bdy_v = f_all[:, mask_bdy]
    h_bdy_v = h_all[:, mask_bdy]
    dzdx_safe = torch.clamp(dz_bdy, max=-1e-10)
    xp_bdy = 1.0 / dzdx_safe

    sqrt_h = torch.sqrt(h_bdy_v)
    conn_z = torch.sqrt(h_bdy_v * xp_bdy**2 + 1.0 / f_bdy_v)
    disc_z = 1.0 / torch.sqrt(f_bdy_v)
    g_z = (sqrt_h / z_bdy**2) * (conn_z - disc_z)

    if use_mc:
        abs_dzdx = torch.abs(dzdx_safe)
        x_bdy_width = l_arr / 2.0 * (1.0 - 1e-8 - u_s)
        I_bdy = x_bdy_width * (g_z * abs_dzdx).mean(dim=1)
    else:
        z_sorted, idx = torch.sort(z_bdy, dim=1)
        g_sorted = torch.gather(g_z, 1, idx)
        I_bdy = torch.trapezoid(g_sorted, z_sorted, dim=1)

    # --- Remainder ---
    if use_mc:
        z_max_bdy = z_bdy.max(dim=1).values
        N_rem = 300
        t_rem = torch.rand(N_rem, dtype=torch.float64) * (1 - 1e-10)
    else:
        z_sorted_for_mid, _ = torch.sort(z_bdy, dim=1)
        z_max_bdy = z_sorted_for_mid[:, -1]
        N_rem = 300
        t_rem = torch.linspace(0, 1 - 1e-10, N_rem, dtype=torch.float64)

    dz_span = z_h - z_max_bdy
    z_rem = z_h - dz_span.unsqueeze(1) * (1 - t_rem.unsqueeze(0))**2
    dz_dt = 2 * dz_span.unsqueeze(1) * (1 - t_rem.unsqueeze(0))
    f_rem = torch.clamp(v_model.forward_f(z_rem.reshape(-1)).reshape(B, N_rem), min=1e-12)
    h_rem = v_model.forward_h(z_rem.reshape(-1)).reshape(B, N_rem)
    integ_rem = torch.sqrt(h_rem) * dz_dt / (z_rem**2 * torch.sqrt(f_rem))

    if use_mc:
        I_rem = (1 - 1e-10) * integ_rem.mean(dim=1)
    else:
        I_rem = torch.trapezoid(integ_rem, t_rem.unsqueeze(0).expand(B, -1), dim=1)

    return list(I_interior + I_bdy - I_rem)


# =============================================================================
# Training
# =============================================================================
def train_inverse_gr_d3(S_data_l, S_data_A, Q, n_epochs=500000,
                        lr_L=1e-4, lr_f=5e-4, lr_h=5e-4,
                        n_grid=4000, print_every=2000,
                        checkpoint_every=50000, seed=42,
                        lambda_s=1.0, resume_from=None,
                        use_mc=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(base_dir, 'data', 'inverse_gr_d3_v6', 'checkpoints')
    plot_dir = os.path.join(base_dir, 'plots', 'inverse_gr_d3_v6')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    l_min, l_max = float(S_data_l.min()), float(S_data_l.max())
    S_interp = interp1d(S_data_l, S_data_A, kind='cubic',
                        fill_value='extrapolate')

    # Known thermal entropy: s * 4G_N = h(z_h) = (1+Q)^{3/2}
    s_target = (1.0 + Q) ** 1.5  # = h(z_h), in units where 4G_N = 1

    l_model = LModel(hidden=32, depth=2, d=D_BOUNDARY_D3)
    v_model = VModelGR_d3(z_h=Z_H, h_h=s_target, hidden=20, depth=2)

    start_epoch = 1
    if resume_from is not None:
        ckpt = torch.load(resume_from, weights_only=False)
        l_model.load_state_dict(ckpt['l_model'])
        v_model.load_state_dict(ckpt['v_model'])
        start_epoch = ckpt['epoch'] + 1
        logger.info(f"  Resumed from {resume_from} (epoch {ckpt['epoch']})")

    opt_L = optim.Adam(l_model.parameters(), lr=lr_L)
    # Single optimizer for all V-model params (D_f, D_h, and shared a)
    opt_V = optim.Adam(v_model.parameters(), lr=lr_f)

    n_L = sum(p.numel() for p in l_model.parameters())
    n_V = sum(p.numel() for p in v_model.parameters())
    logger.info(f"  L: {n_L}, V-model: {n_V} params (incl. shared a)")
    logger.info(f"  lr_L={lr_L}, lr_V={lr_f}, epochs={n_epochs}")
    logger.info(f"  Q={Q}, s_target(=h_h)={s_target:.6f}, lambda_s={lambda_s}")

    loss_hist = []
    t0 = time.time()

    # Exact metric for diagnostics
    from ode_benchmark_gr_d3 import f_exact, h_exact

    # Exact V-model for on-the-fly targets (same integrator, no bias)
    exact_v = ExactVModel_d3(Q)

    batch_size = 2

    for epoch in range(start_epoch, n_epochs + 1):

        if epoch % 3 == 0:
            # Update V-model (D_f, D_h, shared a) on data loss + entropy penalty
            opt_V.zero_grad()

            l_val = float(np.random.uniform(l_min, l_max))
            S_target = float(S_interp(l_val))
            x_np = make_half_grid(l_val, n_grid, x_s_frac=X_S_FRAC)
            A_reg = rt_area_V_gr_d3(l_model, v_model, x_np, l_val,
                                     use_mc=use_mc)
            data_loss = 10000.0 * (A_reg - S_target)**2

            # Thermal entropy penalty: h(z_h) = (1+Q)^{3/2}
            h_zh = v_model.forward_h(
                torch.tensor([Z_H], dtype=torch.float64))
            entropy_loss = lambda_s * (h_zh[0] - s_target)**2

            # Temperature penalty: |f'(z_h)| = 4*pi*T = 3*sqrt(1+Q)
            dz = 1e-4
            z_hz = torch.tensor([Z_H - dz, Z_H], dtype=torch.float64)
            f_hz = v_model.forward_f(z_hz)
            fp_zh = (f_hz[1] - f_hz[0]) / dz
            T_target = 3.0 * np.sqrt(1.0 + Q) / (4.0 * np.pi)
            temp_loss = lambda_s * ((-fp_zh) / (4.0 * np.pi) - T_target)**2

            loss = data_loss + entropy_loss + temp_loss
            loss.backward()
            opt_V.step()
        else:
            # Update L: minimize area (single l)
            l_val = float(np.random.uniform(l_min, l_max))
            x_np = make_half_grid(l_val, n_grid, x_s_frac=X_S_FRAC)
            A_reg = rt_area_V_gr_d3(l_model, v_model, x_np, l_val,
                                     use_mc=use_mc)
            S_target = float(S_interp(l_val))  # for logging only
            opt_L.zero_grad()
            loss = A_reg
            loss.backward()
            opt_L.step()

        loss_hist.append(loss.item())

        if epoch % print_every == 0 or epoch == 1:
            z_pts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            z_test = torch.tensor(z_pts, dtype=torch.float64)
            with torch.no_grad():
                f_test = v_model.forward_f(z_test).numpy()
                h_test = v_model.forward_h(z_test).numpy()
            f_ex = np.array([f_exact(z, Q) for z in z_pts])
            h_ex = np.array([h_exact(z, Q) for z in z_pts])

            elapsed = time.time() - t0
            rate = epoch / elapsed if elapsed > 0 else 0
            logger.info(
                f"  ep {epoch:7d}/{n_epochs}  l={l_val:.3f}  "
                f"|r|={abs(A_reg.item()-S_target):.4f}  "
                f"[{elapsed/60:.1f}min {rate:.0f}ep/s]"
            )
            f_line = "    f: " + ", ".join(f"f({z})={v:.3f}({e:.3f})" for z, v, e in zip(z_pts, f_test, f_ex))
            h_line = "    h: " + ", ".join(f"h({z})={v:.3f}({e:.3f})" for z, v, e in zip(z_pts, h_test, h_ex))
            logger.info(f_line)
            logger.info(h_line)
            # Shared parameter a, h(z_h), and temperature
            a_val = v_model.a.item()
            with torch.no_grad():
                h_zh = v_model.forward_h(
                    torch.tensor([Z_H], dtype=torch.float64)).item()
                z_hz = torch.tensor([Z_H - 1e-4, Z_H], dtype=torch.float64)
                f_hz = v_model.forward_f(z_hz).numpy()
                T_nn = abs(f_hz[1] - f_hz[0]) / 1e-4 / (4 * np.pi)
            T_ex = 3.0 * np.sqrt(1.0 + Q) / (4.0 * np.pi)
            logger.info(
                f"    a={a_val:.4f}(exact={1.5*Q:.4f}) "
                f"h(z_h)={h_zh:.4f}(exact={s_target:.4f}) "
                f"T={T_nn:.4f}(exact={T_ex:.4f})"
            )

        if epoch % print_every == 0:
            # Snapshot plot (overwrites same file)
            _save_snapshot_d3(v_model, l_model, S_interp, (l_min, l_max),
                              epoch, Q, plot_dir)

        if epoch % checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'l_model': l_model.state_dict(),
                'v_model': v_model.state_dict(),
            }, os.path.join(ckpt_dir, f"ep{epoch:06d}.pt"))
            logger.info(f"  >> Checkpoint saved: ep {epoch}")

    t_train = time.time() - t0
    logger.info(f"  Training done: {t_train/60:.1f} min")
    return l_model, v_model, loss_hist


def _save_snapshot_d3(v_model, l_model, S_interp, l_range, epoch, Q, plot_dir):
    """Save 4-panel snapshot: f, h, S_EE, errors. Overwrites same file."""
    from ode_benchmark_gr_d3 import f_exact, h_exact

    z_eval = np.linspace(0, 1, 200)
    z_t = torch.tensor(z_eval, dtype=torch.float64)
    with torch.no_grad():
        f_pred = v_model.forward_f(z_t).numpy()
        h_pred = v_model.forward_h(z_t).numpy()
    f_ex = np.array([f_exact(z, Q) for z in z_eval])
    h_ex = np.array([h_exact(z, Q) for z in z_eval])

    l_min, l_max = l_range
    l_arr = np.linspace(l_min, l_max, 25)
    A_pred, A_ref = [], []
    l_model.eval(); v_model.eval()
    for lv in l_arr:
        x_np = make_half_grid(lv, 2000, x_s_frac=X_S_FRAC)
        A_pred.append(rt_area_V_gr_d3(l_model, v_model, x_np, lv).item())
        A_ref.append(float(S_interp(lv)))
    l_model.train(); v_model.train()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(z_eval, f_ex, 'r-', lw=1.5, label='Exact')
    axes[0, 0].plot(z_eval, f_pred, 'b--', lw=1.5, label='Learned')
    axes[0, 0].set_xlabel(r'$z$'); axes[0, 0].set_ylabel(r'$f(z)$')
    axes[0, 0].set_title(f'Blackening factor (Q={Q}, ep {epoch})')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(z_eval, h_ex, 'r-', lw=1.5, label='Exact')
    axes[0, 1].plot(z_eval, h_pred, 'b--', lw=1.5, label='Learned')
    axes[0, 1].set_xlabel(r'$z$'); axes[0, 1].set_ylabel(r'$h(z)$')
    axes[0, 1].set_title(f'Spatial warp factor (Q={Q}, ep {epoch})')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(l_arr, A_ref, 'r-', lw=1.2, label='Data')
    axes[1, 0].plot(l_arr, A_pred, 'b--', lw=1.0, label='Learned')
    axes[1, 0].set_xlabel(r'$l$'); axes[1, 0].set_ylabel(r'$A_{\mathrm{reg}}$')
    axes[1, 0].set_title(f'$S_{{EE}}(l)$ ep {epoch}')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    mask = z_eval < 0.95
    f_err = np.abs(f_pred[mask] - f_ex[mask])
    h_err = np.abs(h_pred[mask] - h_ex[mask])
    axes[1, 1].plot(z_eval[mask], f_err, 'b-', lw=1.2,
                    label=r'$|f_{\rm learned}-f_{\rm exact}|$')
    axes[1, 1].plot(z_eval[mask], h_err, 'r-', lw=1.2,
                    label=r'$|h_{\rm learned}-h_{\rm exact}|$')
    axes[1, 1].set_xlabel(r'$z$'); axes[1, 1].set_ylabel('Absolute error')
    axes[1, 1].set_title(f'Metric recovery errors (ep {epoch})')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_yscale('log')

    # RT profiles for multiple l values
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d as _interp1d
    try:
        ode_data = np.load(os.path.join(os.path.dirname(plot_dir), '..', 'data',
                                         'ode_benchmark_gr_d3.npz'))
        _l_to_zs = _interp1d(ode_data['l_of_zstar'], ode_data['z_star_grid'],
                              kind='cubic', fill_value='extrapolate')
    except Exception:
        _l_to_zs = None

    colors = ['b', 'g', 'm', 'c', 'orange']
    l_profiles = np.linspace(l_min + 0.02, l_max - 0.02, 5)
    l_model.eval()
    for il, lv in enumerate(l_profiles):
        x_prof = make_half_grid(lv, 2000, x_s_frac=X_S_FRAC)
        x_t = torch.tensor(x_prof, dtype=torch.float64).unsqueeze(1)
        with torch.no_grad():
            z_prof = l_model(x_t, lv).squeeze(1).numpy()
        axes[0, 2].plot(x_prof, z_prof, '-', color=colors[il], lw=1.2,
                        label=f'l={lv:.2f}')
        # ODE profile
        if _l_to_zs is not None:
            try:
                zs_v = float(_l_to_zs(lv))
                hs_v = h_exact(zs_v, Q)
                def _rhs(x, y, _zs=zs_v, _hs=hs_v):
                    z = y[0]
                    if z <= EPSILON * 1.01 or z >= _zs:
                        return [0.0]
                    fv = f_exact(z, Q)
                    hv = h_exact(z, Q)
                    if fv <= 0 or hv <= 0:
                        return [0.0]
                    ratio = hv**2 * _zs**4 / (z**4 * _hs**2) - 1.0
                    if ratio <= 0:
                        return [0.0]
                    return [-np.sqrt(fv * hv * ratio)]
                _sol = solve_ivp(_rhs, (0, lv / 2 * 1.05), [zs_v - 1e-10],
                                 method='RK45', rtol=1e-10, atol=1e-12,
                                 dense_output=True)
                z_ode = _sol.sol(np.clip(x_prof, 0, _sol.t[-1] * 0.999))[0]
                axes[0, 2].plot(x_prof, z_ode, '--', color='black', lw=0.8)
            except Exception:
                pass
    l_model.train()
    axes[0, 2].set_xlabel(r'$x$'); axes[0, 2].set_ylabel(r'$z(x)$')
    axes[0, 2].set_title(f'RT profiles ep {epoch} (solid=NN, dashed=ODE)')
    axes[0, 2].legend(fontsize=7); axes[0, 2].invert_yaxis(); axes[0, 2].grid(alpha=0.3)

    # f'(0) and h'(0) evolution (leave blank for now, just show current)
    a_val = v_model.a.item()
    with torch.no_grad():
        h_zh = v_model.forward_h(
            torch.tensor([1.0], dtype=torch.float64)).item()
    axes[1, 2].text(0.5, 0.7, f"a = {a_val:.4f}\nexact = {1.5*Q:.4f}\n"
                    f"h(z_h) = {h_zh:.4f}\nexact = {(1+Q)**1.5:.4f}",
                    transform=axes[1, 2].transAxes, fontsize=14,
                    verticalalignment='top', horizontalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].set_title(f'Parameters (ep {epoch})')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'snapshot_live.png'), dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 65)
    logger.info("Inverse problem (AdS4 GR): learn f(z) and h(z) from S_EE(l)")
    logger.info("=" * 65)

    Q = 1.0
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    ode_path = os.path.join(data_dir, 'ode_benchmark_gr_d3.npz')
    if not os.path.exists(ode_path):
        logger.error(f"ODE benchmark not found: {ode_path}")
        logger.error("Run ode_benchmark_gr_d3.py first!")
        sys.exit(1)

    ode = np.load(ode_path)
    z_star_grid = ode['z_star_grid']
    l_of_zstar = ode['l_of_zstar']
    A_reg_of_zstar = ode['A_reg_of_zstar']
    l_c = float(ode['l_c'][0])

    logger.info(f"  Q={Q}, l_c={l_c:.6f}")

    l_to_zs = interp1d(l_of_zstar, z_star_grid,
                        kind='cubic', fill_value='extrapolate')
    zs_to_Areg = interp1d(z_star_grid, A_reg_of_zstar,
                           kind='cubic', fill_value='extrapolate')

    l_data = np.linspace(0.2, 0.8 * l_c, 50)
    A_data = np.array([float(zs_to_Areg(float(l_to_zs(lv)))) for lv in l_data])
    logger.info(f"  Input: {len(l_data)} pts, l in [{l_data[0]:.3f}, {l_data[-1]:.3f}]")
    logger.info(f"  A_reg range: [{A_data.min():.4f}, {A_data.max():.4f}]")

    # Resume from specific checkpoint (ep400000 — before drift)
    ckpt_dir_resume = os.path.join(data_dir, 'inverse_gr_d3_v6', 'checkpoints')
    resume = os.path.join(ckpt_dir_resume, 'ep400000.pt')

    l_model, v_model, loss_hist = train_inverse_gr_d3(
        l_data, A_data, Q=Q, n_epochs=1000000,
        lr_L=1e-4, lr_f=5e-4, lr_h=5e-4,
        n_grid=4000, print_every=2000, lambda_s=10000.0,
        resume_from=resume, use_mc=False)

    logger.info("  DONE.")


if __name__ == "__main__":
    main()
