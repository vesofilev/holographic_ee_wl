# ASSERT_CONVENTION: inverse problem — learn f(z) from S_EE(l) data
# ASSERT_CONVENTION: follows pinn_d7 alternating optimization (Section 4, arXiv:2506.20115)
"""
Inverse problem: jointly learn the RT surface z(x,l) and the blackening
factor f(z) from entanglement entropy data S_EE(l).

Adapted directly from pinn_d7/train_inverse.py and pinn_d7/models.py.
"""
import os
import sys
import json
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("inverse_training.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import make_half_grid, load_ode_benchmark, Z_H, D_BOUNDARY
from ann_forward import EPSILON, X_S_FRAC

rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


# =============================================================================
# L-model: conditional surface z(x, l)
# Copied from pinn_d7 LNetworkM pattern: 2 inputs, hard-wired BC
# =============================================================================
class LModel(nn.Module):
    """z(x, l) = eps + (l^2/4 - x^2)^{1/d} * softplus(g([x^2, l]; W))"""
    def __init__(self, hidden=16, depth=2, d=D_BOUNDARY, epsilon=EPSILON):
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

    def forward(self, x, l_val):
        """x: (N,1), l_val: scalar or (N,1). Returns z: (N,1).
        Uses sigmoid to bound z < z_h, preventing horizon crossing."""
        if not isinstance(l_val, torch.Tensor):
            l_t = torch.full_like(x, l_val)
        else:
            l_t = l_val * torch.ones_like(x)
        x_sq = (x / (l_t / 2.0)) ** 2
        inp = torch.cat([x_sq, l_t], dim=1)
        g = self.core(inp)
        prefactor = (l_t / 2.0) ** 2 - x ** 2
        prefactor = torch.clamp(prefactor, min=0.0)
        return self.epsilon + prefactor ** (1.0 / self.d) * torch.nn.functional.softplus(g)


# =============================================================================
# V-model: blackening factor f(z)
# Copied from pinn_d7 VNetwork: normalize between endpoints
# =============================================================================
class VModel(nn.Module):
    """f(z) via normalization: f = (g(z) - g(z_h)) / (g(0) - g(z_h)).

    Exactly the pinn_d7 VNetwork pattern:
      - f(0) = 1 and f(z_h) = 0 by construction
      - No sigmoid => no saturation => gradients always flow
      - Network output g(z) is unconstrained
    """
    def __init__(self, z_h=Z_H, hidden=16, depth=2):
        super().__init__()
        self.z_h = z_h
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.core = nn.Sequential(*layers).double()
        # Init: bias=1 on final layer so g(z) starts ~1 everywhere
        for m in self.core:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.05)
                nn.init.zeros_(m.bias)
        self.core[-1].bias.data.fill_(1.0)

    def forward(self, z):
        """z: any shape. Returns f(z) with f(0)=1, f(z_h)=0."""
        if z.dim() == 0:
            z = z.unsqueeze(0)
        orig_shape = z.shape
        zn = (z / self.z_h).reshape(-1, 1)

        g = self.core(zn)
        g0 = self.core(torch.zeros(1, 1, dtype=z.dtype, device=z.device))
        g1 = self.core(torch.ones(1, 1, dtype=z.dtype, device=z.device))

        f = (g - g1) / (g0 - g1 + 1e-8)
        return f.squeeze(-1).reshape(orig_shape)


# =============================================================================
# Area functional with learned f(z)
# =============================================================================
def rt_area_V(l_model, v_model, x_grid_np, l_val, x_s_frac=X_S_FRAC,
              epsilon=EPSILON, z_h=Z_H):
    """Compute A_reg_half using learned f(z) from v_model.

    Uses the same hybrid x/z regularization as the forward problem.
    """
    x_grid = torch.tensor(x_grid_np, dtype=torch.float64)
    x_t = x_grid.unsqueeze(1).clone().requires_grad_(True)

    z = l_model(x_t, l_val)
    dzdx = torch.autograd.grad(z.sum(), x_t, create_graph=True)[0]

    z_flat = z.squeeze(1)
    dzdx_flat = dzdx.squeeze(1)
    x_flat = x_grid

    # Learned f(z)
    f_vals = v_model(z_flat)

    x_s = x_s_frac * l_val

    # --- Interior ---
    mask_int = x_flat <= x_s + 1e-14
    x_int = x_flat[mask_int]
    z_int = z_flat[mask_int]
    dz_int = dzdx_flat[mask_int]
    f_int = torch.clamp(f_vals[mask_int], min=1e-12)
    integ_int = (1.0 / z_int ** 3) * torch.sqrt(1.0 + dz_int ** 2 / f_int)
    I_interior = torch.trapezoid(integ_int, x_int)

    # --- Boundary (z-parameterization) ---
    mask_bdy = x_flat > x_s + 1e-14
    z_bdy = z_flat[mask_bdy]
    dzdx_bdy = dzdx_flat[mask_bdy]
    f_bdy = torch.clamp(f_vals[mask_bdy], min=1e-12)

    dzdx_safe = torch.clamp(dzdx_bdy, max=-1e-10)
    xp_bdy = 1.0 / dzdx_safe

    z_sorted, idx = torch.sort(z_bdy)
    xp_sorted = xp_bdy[idx]
    f_sorted = f_bdy[idx]

    conn = torch.sqrt(xp_sorted ** 2 + 1.0 / f_sorted)
    disc = 1.0 / torch.sqrt(f_sorted)
    sub_integ = (1.0 / z_sorted ** 3) * (conn - disc)
    I_bdy = torch.trapezoid(sub_integ, z_sorted)

    # --- Remainder: change of variables to remove 1/sqrt(f) singularity ---
    # z(t) = z_h - (z_h - z_mid)*(1-t)^2, dz/dt = 2*(z_h - z_mid)*(1-t)
    # The (1-t) Jacobian cancels the 1/sqrt(z_h - z) divergence in 1/sqrt(f).
    z_mid = z_sorted[-1]  # keep in graph for L-model gradients
    N_rem = 300
    t_rem = torch.linspace(0, 1 - 1e-10, N_rem, dtype=torch.float64)
    dz_span = z_h - z_mid
    z_rem = z_h - dz_span * (1 - t_rem) ** 2
    dz_dt = 2 * dz_span * (1 - t_rem)
    f_rem = torch.clamp(v_model(z_rem), min=1e-12)
    integ_rem = dz_dt / (z_rem ** 3 * torch.sqrt(f_rem))
    I_rem = torch.trapezoid(integ_rem, t_rem)

    return I_interior + I_bdy - I_rem


# =============================================================================
# Training (following pinn_d7/train_inverse.py exactly)
# =============================================================================
def train_inverse(S_data_l, S_data_A, n_epochs=500000,
                  lr_L=1e-4, lr_V=5e-4,  # V learns faster (like pinn_d7)
                  n_grid=4000, print_every=2000, checkpoint_every=50000,
                  plot_every=10000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs("data/inverse/checkpoints", exist_ok=True)
    os.makedirs("plots/inverse", exist_ok=True)

    l_min, l_max = float(S_data_l.min()), float(S_data_l.max())
    from scipy.interpolate import interp1d
    S_interp = interp1d(S_data_l, S_data_A, kind='cubic', fill_value='extrapolate')

    l_model = LModel(hidden=16, depth=2)
    v_model = VModel(z_h=Z_H, hidden=16, depth=2)

    opt_L = optim.Adam(l_model.parameters(), lr=lr_L)
    opt_V = optim.Adam(v_model.parameters(), lr=lr_V)

    n_L = sum(p.numel() for p in l_model.parameters())
    n_V = sum(p.numel() for p in v_model.parameters())
    logger.info(f"  L-model: {n_L} params, V-model: {n_V} params")
    logger.info(f"  lr_L={lr_L}, lr_V={lr_V}, epochs={n_epochs}")

    loss_hist = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        l_val = float(np.random.uniform(l_min, l_max))
        S_target = float(S_interp(l_val))

        x_np = make_half_grid(l_val, n_grid, x_s_frac=X_S_FRAC)
        A_reg = rt_area_V(l_model, v_model, x_np, l_val)

        if epoch % 2 == 0:
            # Update V (data loss, amplified by 100 like pinn_d7)
            opt_V.zero_grad()
            loss = 100.0 * (A_reg - S_target) ** 2
            loss.backward()
            opt_V.step()
        else:
            # Update L (physical loss = minimize area)
            opt_L.zero_grad()
            loss = A_reg
            loss.backward()
            opt_L.step()

        loss_hist.append(loss.item())

        if epoch % print_every == 0 or epoch == 1:
            z_test = torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
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

        if epoch % plot_every == 0:
            _save_snapshot(v_model, l_model, S_interp, (l_min, l_max), epoch)

        if epoch % checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'l_model': l_model.state_dict(),
                'v_model': v_model.state_dict(),
            }, f"data/inverse/checkpoints/ep{epoch:06d}.pt")
            logger.info(f"  >> Checkpoint saved: ep {epoch}")

    t_train = time.time() - t0
    logger.info(f"  Training done: {t_train/60:.1f} min")
    return l_model, v_model, loss_hist


def _save_snapshot(v_model, l_model, S_interp, l_range, epoch):
    """Save diagnostic f(z) + S_EE(l) snapshot."""
    z_eval = np.linspace(0, Z_H, 200)
    z_t = torch.tensor(z_eval, dtype=torch.float64)
    with torch.no_grad():
        f_pred = v_model(z_t).numpy()
    f_exact = 1.0 - z_eval ** 4

    l_min, l_max = l_range
    l_arr = np.linspace(l_min, l_max, 25)
    A_pred, A_ref = [], []
    l_model.eval()
    v_model.eval()
    for lv in l_arr:
        x_np = make_half_grid(lv, 2000, x_s_frac=X_S_FRAC)
        A_pred.append(rt_area_V(l_model, v_model, x_np, lv).item())
        A_ref.append(float(S_interp(lv)))
    l_model.train()
    v_model.train()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(z_eval, f_exact, 'r-', lw=1.5, label=r'Exact $1-z^4$')
    ax1.plot(z_eval, f_pred, 'b--', lw=1.5, label='Learned')
    ax1.set_xlabel(r'$z/z_h$'); ax1.set_ylabel(r'$f(z)$')
    ax1.set_title(f'epoch {epoch}'); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(l_arr, A_ref, 'r-', lw=1.2, label='Data')
    ax2.plot(l_arr, A_pred, 'b--', lw=1.0, label='ANN')
    ax2.set_xlabel(r'$l/z_h$'); ax2.set_ylabel(r'$A_{\mathrm{reg}}$')
    ax2.set_title(f'$S_{{EE}}(l)$ ep {epoch}'); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/inverse/snapshot_ep{epoch:06d}.png", dpi=150)
    plt.close()
    logger.info(f"  >> Snapshot: plots/inverse/snapshot_ep{epoch:06d}.png")


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 65)
    logger.info("Inverse problem: learn f(z) from S_EE(l) data")
    logger.info("=" * 65)

    ode = load_ode_benchmark("data/ode_benchmark.npz")
    l_c = ode['l_c']
    from scipy.interpolate import interp1d
    l_to_zs = interp1d(ode['l_of_zstar'], ode['z_star_grid'],
                        kind='cubic', fill_value='extrapolate')
    zs_to_Areg = interp1d(ode['z_star_grid'], ode['A_reg_of_zstar'],
                           kind='cubic', fill_value='extrapolate')

    # Input data
    l_data = np.linspace(0.15, 0.75 * l_c, 50)
    A_data = np.array([float(zs_to_Areg(float(l_to_zs(lv)))) for lv in l_data])
    logger.info(f"  Input data: {len(l_data)} pts, l in [{l_data[0]:.3f}, {l_data[-1]:.3f}]")

    l_model, v_model, loss_hist = train_inverse(
        l_data, A_data, n_epochs=500000, lr_L=1e-4, lr_V=5e-4,
        n_grid=4000, print_every=10000)

    # --- Evaluate ---
    os.makedirs("data/inverse", exist_ok=True)

    z_eval = np.linspace(0, Z_H, 200)
    z_t = torch.tensor(z_eval, dtype=torch.float64)
    f_learned = v_model(z_t).detach().numpy()
    f_exact = 1.0 - z_eval ** 4

    mask = z_eval < 0.95
    rel_err = np.abs(f_learned[mask] - f_exact[mask]) / np.maximum(np.abs(f_exact[mask]), 1e-10)
    logger.info(f"  f(z) error (z<0.95): max={rel_err.max():.4f}, mean={rel_err.mean():.4f}")

    # Learned S_EE(l)
    l_eval = np.linspace(l_data[0], l_data[-1], 80)
    A_ann = []
    l_model.eval()
    v_model.eval()
    for lv in l_eval:
        x_np = make_half_grid(lv, 4000, x_s_frac=X_S_FRAC)
        A = rt_area_V(l_model, v_model, x_np, lv).item()
        A_ann.append(A)
    A_ode_eval = [float(zs_to_Areg(float(l_to_zs(lv)))) for lv in l_eval]

    np.savez("data/inverse/results.npz",
             z=z_eval, f_learned=f_learned, f_exact=f_exact,
             l_eval=l_eval, A_ann=A_ann, A_ode=A_ode_eval)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(z_eval, f_exact, 'r-', lw=1.5, label=r'Exact: $f(z) = 1 - z^4$')
    ax1.plot(z_eval, f_learned, 'b--', lw=1.5, label='Learned $f(z)$')
    ax1.set_xlabel(r'$z / z_h$')
    ax1.set_ylabel(r'$f(z)$')
    ax1.set_title('Recovered blackening factor')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(l_eval, A_ode_eval, 'r-', lw=1.2, label='Input data (ODE)')
    ax2.plot(l_eval, A_ann, 'b--', lw=1.0, label='Learned $S_{EE}(l)$')
    ax2.set_xlabel(r'$l / z_h$')
    ax2.set_ylabel(r'$A_{\mathrm{reg,half}}$')
    ax2.set_title('Entanglement entropy: learned vs data')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/inverse_metric.pdf")
    plt.close()
    logger.info("  Saved figures/inverse_metric.pdf")
    logger.info("  DONE.")


if __name__ == "__main__":
    main()
