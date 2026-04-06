# ASSERT_CONVENTION: natural_units=natural, metric_signature=mostly_minus,
#   coordinate_system=fefferman_graham_z, z_h=1, d=4, L=1
# ASSERT_CONVENTION: f(z)=1-z^4, float64 for area
# ASSERT_CONVENTION: area = V_2 * int (1/z^3) sqrt(1 + z'^2/f(z)) dx
# ASSERT_CONVENTION: loss = A_reg_half via hybrid x/z regularization (NO EOM)
# ASSERT_CONVENTION: bc_encoding z(x) = eps + ((l/2)^2 - x^2)^{1/d} * softplus(g_NN(x^2))
"""
ANN area-as-loss for RT surfaces with UV-finite regularization.

The regularized area is computed via a hybrid parameterization:
  - Interior [0, x_s]: standard x-integral (finite, no divergence)
  - Boundary [x_s, l/2]: re-parameterized in z, integrand-level subtraction
    of disconnected surface makes the integrand pointwise UV-finite
  - Remainder: int_{z_mid}^{z_h} dz/(z^3 sqrt(f)) (finite, no UV region)

No equation of motion is used at any stage.
"""

import os
import sys
import json
import time
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import blackening_factor, make_half_grid, load_ode_benchmark, Z_H, D_BOUNDARY

EPSILON = 1e-4       # UV cutoff (can be small thanks to proper regularization)
X_S_FRAC = 1.0 / 5  # split at x_s = l/5


# =============================================================================
# Neural Network
# =============================================================================
class RTSurfaceNet(nn.Module):
    """MLP for g_NN(x^2) in the power-law BC encoding."""

    def __init__(self, hidden_dim=64, n_hidden=3, l=1.0, z_star_guess=0.5,
                 epsilon=EPSILON):
        super().__init__()
        self.l_half = l / 2.0
        self.d = D_BOUNDARY

        # Compute initial bias so that z(0) ~ z_star_guess at initialisation.
        # z(0) = eps + (l/2)^{2/d} * softplus(g_bias + net(0))
        # With small-init net(0) ~ 0, need softplus(g_bias) ~ target.
        prefactor_0 = self.l_half ** (2.0 / self.d)
        target_sp = max((z_star_guess - epsilon) / prefactor_0, 1e-3)
        # inverse softplus: x = log(exp(y) - 1)
        self.g_bias = float(np.log(np.expm1(target_sp)))

        layers = [nn.Linear(1, hidden_dim, dtype=torch.float64), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64),
                       nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1, dtype=torch.float64))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                fan = m.weight.size(0) + m.weight.size(1)
                nn.init.normal_(m.weight, 0.0, np.sqrt(2.0 / fan) * 0.3)
                nn.init.zeros_(m.bias)
        final = self.net[-1]
        nn.init.normal_(final.weight, 0.0, 0.01)
        nn.init.zeros_(final.bias)

    def forward(self, x):
        x_in = ((x / self.l_half) ** 2).unsqueeze(-1)
        return self.g_bias + self.net(x_in).squeeze(-1)


# =============================================================================
# BC Encoding  (power-law prefactor)
# =============================================================================
def encode_bc(g_nn, x, l, d=D_BOUNDARY, epsilon=EPSILON):
    """z(x) = eps + ((l/2)^2 - x^2)^{1/d} * softplus(g_NN(x^2)).

    The exponent 1/d gives z ~ (l/2-x)^{1/d} near the boundary,
    so z'(x) -> -inf, ensuring UV cancellation in the regularized area.
    """
    prefactor = (l / 2.0) ** 2 - x ** 2
    prefactor = torch.clamp(prefactor, min=0.0)
    return epsilon + prefactor ** (1.0 / d) * torch.nn.functional.softplus(g_nn)


# =============================================================================
# Custom autograd: remainder integral  int_{z_mid}^{z_h} dz/(z^3 sqrt(f))
# =============================================================================
class _IntegralRemainder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_mid, z_h_val):
        from scipy.integrate import quad
        zm = z_mid.detach().item()

        def integ(z):
            fv = 1.0 - (z / z_h_val) ** 4
            return 1.0 / (z ** 3 * np.sqrt(fv)) if fv > 0 else 0.0

        val, _ = quad(integ, zm, z_h_val * (1 - 1e-12),
                      epsabs=1e-12, epsrel=1e-12, limit=200)
        ctx.save_for_backward(z_mid)
        ctx.z_h_val = z_h_val
        return z_mid.new_tensor(val)

    @staticmethod
    def backward(ctx, grad_output):
        z_mid, = ctx.saved_tensors
        f_zm = 1.0 - (z_mid / ctx.z_h_val) ** 4
        # d/d(z_mid) int_{z_mid}^{z_h} g dz = -g(z_mid)
        grad = -1.0 / (z_mid ** 3 * torch.sqrt(torch.clamp(f_zm, min=1e-12)))
        return grad_output * grad, None


integral_remainder = _IntegralRemainder.apply


# =============================================================================
# UV-finite regularized loss (Eq. 8 in draft)
# =============================================================================
def compute_regularized_loss(z, dzdx, x_grid, l,
                             x_s_frac=X_S_FRAC, epsilon=EPSILON, z_h=Z_H):
    """
    A_reg_half via hybrid x/z parameterization.

    Returns scalar loss = I_interior + I_boundary_sub - I_remainder.
    All three integrals are individually finite.
    """
    x_s = x_s_frac * l

    # ---------- Interior: [0, x_s] in x ----------
    mask_int = x_grid <= x_s + 1e-14
    x_int, z_int, dz_int = x_grid[mask_int], z[mask_int], dzdx[mask_int]
    f_int = torch.clamp(blackening_factor(z_int), min=1e-12)
    integ_int = (1.0 / z_int ** 3) * torch.sqrt(1.0 + dz_int ** 2 / f_int)
    I_interior = torch.trapezoid(integ_int, x_int)

    # ---------- Boundary: (x_s, l/2] -> z-parameterization ----------
    mask_bdy = x_grid > x_s + 1e-14
    z_bdy = z[mask_bdy]
    dzdx_bdy = dzdx[mask_bdy]

    # z' < 0 in boundary; clamp away from zero for safe division
    dzdx_safe = torch.clamp(dzdx_bdy, max=-1e-10)
    xp_bdy = 1.0 / dzdx_safe          # x'(z) = 1/z'(x), negative

    f_bdy = torch.clamp(blackening_factor(z_bdy), min=1e-12)

    # Sort by z ascending (eps -> z_mid)
    z_sorted, idx = torch.sort(z_bdy)
    xp_sorted = xp_bdy[idx]
    f_sorted = f_bdy[idx]

    # Subtracted integrand: (1/z^3)[ sqrt(x'^2 + 1/f) - 1/sqrt(f) ]
    conn = torch.sqrt(xp_sorted ** 2 + 1.0 / f_sorted)
    disc = 1.0 / torch.sqrt(f_sorted)
    sub_integ = (1.0 / z_sorted ** 3) * (conn - disc)
    I_boundary_sub = torch.trapezoid(sub_integ, z_sorted)

    # ---------- Remainder: int_{z_mid}^{z_h} dz/(z^3 sqrt(f)) ----------
    z_mid = z_sorted[-1]
    I_remainder = integral_remainder(z_mid, z_h)

    return I_interior + I_boundary_sub - I_remainder


# =============================================================================
# Reference A_reg from ODE (Eq. 6, uses EOM — for VALIDATION only)
# =============================================================================
def compute_A_reg_from_zstar(z_star, z_h=Z_H):
    """UV-finite A_reg using first-integral formula. NOT used in training."""
    from scipy.integrate import quad

    def integ1(z):
        if z < 1e-15:
            return 0.0
        r6 = (z / z_star) ** 6
        if r6 >= 1.0 - 1e-15:
            return 0.0
        fv = 1.0 - (z / z_h) ** 4
        return (1.0 / np.sqrt(1.0 - r6) - 1.0) / (z ** 3 * np.sqrt(fv)) if fv > 0 else 0.0

    I1, _ = quad(integ1, 0, z_star * (1 - 1e-12),
                 epsabs=1e-12, epsrel=1e-12, limit=200, points=[z_star * 0.5])

    def integ2(z):
        fv = 1.0 - (z / z_h) ** 4
        return 1.0 / (z ** 3 * np.sqrt(fv)) if fv > 0 else 0.0

    I2, _ = quad(integ2, z_star, z_h * (1 - 1e-12),
                 epsabs=1e-12, epsrel=1e-12, limit=200)
    return I1 - I2


# =============================================================================
# Diagnostics
# =============================================================================
def check_first_integral(z, dzdx, z_star):
    """H = 1/(z^3 sqrt(1+z'^2/f)) should equal 1/z_*^3 on the true solution."""
    f_z = np.maximum(1.0 - z ** 4, 1e-30)
    H = 1.0 / (z ** 3 * np.sqrt(1.0 + dzdx ** 2 / f_z))
    H_exp = 1.0 / z_star ** 3
    interior = z > 2.0 * EPSILON
    if np.sum(interior) == 0:
        return np.nan
    return np.max(np.abs(H[interior] - H_exp)) / H_exp


def reconstruct_ode_profile(z_star, l, x_grid, epsilon=EPSILON):
    """Reconstruct ODE z(x) by integrating from the turning point."""
    from scipy.integrate import solve_ivp

    def rhs(x, y):
        z = y[0]
        if z <= epsilon * 1.01 or z >= z_star:
            return [0.0]
        r = (z_star / z) ** 6 - 1.0
        if r <= 0:
            return [0.0]
        fv = 1.0 - z ** 4
        if fv <= 0:
            return [0.0]
        return [-np.sqrt(fv * r)]

    def hit_bc(x, y):
        return y[0] - epsilon * 1.1
    hit_bc.terminal = True
    hit_bc.direction = -1

    sol = solve_ivp(rhs, (0, l / 2 * 1.05), [z_star - 1e-10],
                    method='RK45', rtol=1e-12, atol=1e-14,
                    events=hit_bc, dense_output=True, max_step=l / 2000)
    x_max = sol.t[-1]
    return sol.sol(np.clip(x_grid, 0, x_max * 0.999))[0]


def save_progress_plot(x_np, z_np, z_ode, loss_hist, epoch, l, plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Profile
    axes[0].plot(x_np, z_np, 'b-', lw=1.2, label='ANN')
    axes[0].plot(x_np, z_ode, 'r--', lw=1.0, label='ODE')
    axes[0].set(xlabel='x', ylabel='z(x)',
                title=f'l={l:.4f}  epoch {epoch}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].semilogy(np.abs(loss_hist), 'k-', lw=0.4)
    axes[1].set(xlabel='epoch', ylabel='|Loss|', title='Loss history')
    axes[1].grid(alpha=0.3)

    # Error profile
    axes[2].semilogy(x_np, np.abs(z_np - z_ode) + 1e-16, 'g-', lw=0.6)
    axes[2].set(xlabel='x', ylabel='|z_ANN - z_ODE|', title='Profile error')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/progress_l_{l:.3f}_ep{epoch:06d}.png", dpi=120)
    plt.close(fig)


# =============================================================================
# Training
# =============================================================================
def train_single_l(l, z_star_ode, n_epochs=50000, lr=1e-4,
                   n_grid=8000, epsilon=EPSILON, seed=None,
                   print_every=200, plot_every=5000):
    """Train one strip width with the UV-finite regularized loss."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    x_np = make_half_grid(l, n_grid, x_s_frac=X_S_FRAC)
    x_grid = torch.tensor(x_np, dtype=torch.float64)
    z_star_guess = min(l / (2.0 * 0.4312), 0.9 * Z_H)

    net = RTSurfaceNet(hidden_dim=20, n_hidden=2, l=l,
                       z_star_guess=z_star_guess, epsilon=epsilon)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    loss_history = []
    best_loss = float('inf')
    best_state = None
    t0 = time.time()

    # ODE reference profile on the same half-grid
    z_ode = reconstruct_ode_profile(z_star_ode, l, x_np, epsilon=epsilon)

    logger.info(f"  Grid: {n_grid} pts on [0, {l/2:.6f}], "
                f"x_s = {X_S_FRAC*l:.6f}, eps = {epsilon:.1e}")
    logger.info(f"  Initial g_bias = {net.g_bias:.4f}  ->  "
                f"z*(init) ~ {epsilon + (l/2)**(2.0/D_BOUNDARY) * float(np.log1p(np.exp(net.g_bias))):.4f}")

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        x_t = x_grid.clone().requires_grad_(True)
        g = net(x_t)
        z = encode_bc(g, x_t, l, d=D_BOUNDARY, epsilon=epsilon)
        dzdx = torch.autograd.grad(z.sum(), x_t, create_graph=True)[0]

        loss = compute_regularized_loss(z, dzdx, x_grid, l,
                                        x_s_frac=X_S_FRAC,
                                        epsilon=epsilon, z_h=Z_H)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        lv = loss.item()
        loss_history.append(lv)

        if lv < best_loss:
            best_loss = lv
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if epoch % print_every == 0 or epoch == 1:
            zs = z.detach().max().item()
            err_z = abs(zs - z_star_ode) / z_star_ode * 100
            cur_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"  [l={l:.4f}] ep {epoch:6d}  loss={lv:+.8e}  "
                f"z*={zs:.6f} (err {err_z:.3f}%)  "
                f"|grad|={max(p.grad.abs().max().item() for p in net.parameters() if p.grad is not None):.2e}  "
                f"lr={cur_lr:.1e}"
            )

        if epoch % plot_every == 0:
            save_progress_plot(x_np, z.detach().numpy(), z_ode,
                               loss_history, epoch, l)

    # ---- Restore best & final eval ----
    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    x_t = x_grid.clone().requires_grad_(True)
    g = net(x_t)
    z_final = encode_bc(g, x_t, l, d=D_BOUNDARY, epsilon=epsilon)
    dzdx_final = torch.autograd.grad(z_final.sum(), x_t, create_graph=False)[0]

    z_np = z_final.detach().numpy()
    dzdx_np = dzdx_final.detach().numpy()
    z_star = float(z_np.max())

    # A_reg from the network's own regularized loss (half-strip value,
    # matching the ODE benchmark convention which also stores half-values)
    A_reg_ann = compute_regularized_loss(
        z_final, dzdx_final, x_grid, l,
        x_s_frac=X_S_FRAC, epsilon=epsilon, z_h=Z_H
    ).item()

    fi_var = check_first_integral(z_np, dzdx_np, z_star)
    t_train = time.time() - t0

    logger.info(f"  [l={l:.4f}] DONE  z*={z_star:.8f}  "
                f"A_reg(ANN)={A_reg_ann:.10f}  FI_var={fi_var:.2e}  "
                f"time={t_train/60:.1f}min")

    return {
        'z_profile': z_np, 'dzdx_profile': dzdx_np, 'x_grid': x_np,
        'A_reg': A_reg_ann, 'z_star': z_star,
        'loss_history': loss_history, 'training_time': t_train,
        'l': l, 'n_grid': n_grid, 'epsilon': epsilon,
        'first_integral_var': fi_var, 'z_ode': z_ode,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 70)
    logger.info("ANN Area-as-Loss — UV-finite regularization (NO EOM)")
    logger.info("=" * 70)
    logger.info(f"  eps = {EPSILON:.1e},  x_s/l = {X_S_FRAC:.3f},  "
                f"d = {D_BOUNDARY},  z_h = {Z_H}")

    ode_data = load_ode_benchmark("data/ode_benchmark.npz")
    l_c = ode_data['l_c']
    logger.info(f"  l_c = {l_c:.10f}")

    from scipy.interpolate import interp1d
    l_to_zstar = interp1d(ode_data['l_of_zstar'], ode_data['z_star_grid'],
                          kind='cubic', fill_value='extrapolate')
    zstar_to_Areg = interp1d(ode_data['z_star_grid'], ode_data['A_reg_of_zstar'],
                             kind='cubic', fill_value='extrapolate')

    test_ls = [0.3, 0.5 * l_c, 0.6, 0.9 * l_c]
    labels  = ['0.3', '0.5*l_c', '0.6', '0.9*l_c']

    results = []
    os.makedirs("data/ann_single_l", exist_ok=True)

    for i, (l_val, label) in enumerate(zip(test_ls, labels)):
        z_star_ode = float(l_to_zstar(l_val))
        A_reg_ode  = float(zstar_to_Areg(z_star_ode))

        logger.info("\n" + "=" * 65)
        logger.info(f"  Strip l = {l_val:.6f}  ({label})")
        logger.info(f"  ODE target: z* = {z_star_ode:.8f},  A_reg = {A_reg_ode:.10f}")
        logger.info("=" * 65)

        res = train_single_l(
            l=l_val, z_star_ode=z_star_ode,
            n_epochs=5000, lr=5e-5, n_grid=8000,
            seed=42 + i, print_every=200, plot_every=5000,
        )

        # Also compute A_reg via EOM formula for cross-check
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A_reg_eom = compute_A_reg_from_zstar(z_star_ode)

        rel_area = abs(res['A_reg'] - A_reg_ode) / abs(A_reg_ode) if A_reg_ode != 0 else float('nan')
        rel_zs   = abs(res['z_star'] - z_star_ode) / z_star_ode
        prof_err = np.max(np.abs(res['z_profile'] - res['z_ode'])) / z_star_ode

        res.update({
            'A_reg_ode': A_reg_ode, 'A_reg_eom': A_reg_eom,
            'z_star_ode': z_star_ode,
            'rel_area_error': rel_area, 'rel_zstar_error': rel_zs,
            'profile_maxnorm': prof_err, 'label': label,
        })
        results.append(res)

        logger.info(f"  >> z* err = {rel_zs:.4e},  A_reg err = {rel_area:.4e},  "
                    f"profile max|Δ| = {prof_err:.4e}")

        # Save per-l artifacts
        np.save(f"data/ann_single_l/z_profile_l{i}.npy", res['z_profile'])
        np.save(f"data/ann_single_l/loss_history_l{i}.npy",
                np.array(res['loss_history']))
        with open(f"data/ann_single_l/metrics_l{i}.json", 'w') as f:
            json.dump({
                'l': l_val, 'label': label,
                'A_reg_ann': res['A_reg'], 'A_reg_ode': A_reg_ode,
                'A_reg_eom': A_reg_eom,
                'z_star_ann': res['z_star'], 'z_star_ode': z_star_ode,
                'rel_area_error': rel_area, 'rel_zstar_error': rel_zs,
                'profile_maxnorm': prof_err,
                'first_integral_var': res['first_integral_var'],
                'training_time': res['training_time'],
                'epsilon': EPSILON, 'x_s_frac': X_S_FRAC,
            }, f, indent=2)

    # ---- Summary table ----
    logger.info("\n" + "=" * 90)
    logger.info("  FINAL SUMMARY  (UV-finite regularization, eps={:.1e})".format(EPSILON))
    logger.info("=" * 90)
    hdr = (f"{'l':>8} | {'z*_ODE':>10} | {'z*_ANN':>10} | "
           f"{'A_reg_ODE':>12} | {'A_reg_ANN':>12} | "
           f"{'area_err':>10} | {'z*_err':>10} | {'FI_var':>10}")
    logger.info(hdr)
    logger.info("-" * len(hdr))

    all_pass = True
    for r in results:
        ok = r['rel_area_error'] < 0.01
        if not ok:
            all_pass = False
        logger.info(
            f"{r['l']:8.4f} | {r['z_star_ode']:10.6f} | {r['z_star']:10.6f} | "
            f"{r['A_reg_ode']:12.8f} | {r['A_reg']:12.8f} | "
            f"{r['rel_area_error']:10.2e} | {r['rel_zstar_error']:10.2e} | "
            f"{r['first_integral_var']:10.2e}"
        )

    logger.info(f"\n  All < 1%: {'YES' if all_pass else 'NO'}")

    with open("data/ann_single_l/summary.json", 'w') as f:
        json.dump({
            'test_ls': [r['l'] for r in results],
            'z_star_ann': [r['z_star'] for r in results],
            'z_star_ode': [r['z_star_ode'] for r in results],
            'A_reg_ann': [r['A_reg'] for r in results],
            'A_reg_ode': [r['A_reg_ode'] for r in results],
            'rel_area_errors': [r['rel_area_error'] for r in results],
            'epsilon': EPSILON, 'x_s_frac': X_S_FRAC,
            'all_pass': all_pass,
        }, f, indent=2)

    logger.info("  DONE.  Artifacts in data/ann_single_l/,  plots in plots/")


if __name__ == "__main__":
    main()
