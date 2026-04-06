"""
Pretrain f, h, and L networks to match the exact GR solution.

Step 1: Fit f_NN(z) to f_exact(z) via MSE on a z-grid.
Step 2: Fit h_NN(z) to h_exact(z) via MSE on a z-grid.
Step 3: Fit L(x, l) by minimizing area with exact f, h frozen,
        for many l values.
Step 4: Save checkpoint compatible with ann_inverse_gr_d3.py.

Then run ann_inverse_gr_d3.py from this checkpoint to test whether
f'(0) drifts even when starting from the exact solution.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ann_inverse_gr_d3 import (VModelGR_d3, rt_area_V_gr_d3,
                                D_BOUNDARY_D3)
from ann_inverse import LModel
from ode_benchmark_gr_d3 import f_exact, h_exact
from utils import make_half_grid, Z_H
from ann_forward import EPSILON, X_S_FRAC

Q = 1.0
h_h = (1.0 + Q) ** 1.5
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pretrain_f_h(v_model, n_steps=20000, lr=1e-3):
    """Fit f_NN and h_NN to exact functions via MSE on z-grid."""
    z_grid = torch.linspace(0, 1, 200, dtype=torch.float64)
    f_target = torch.tensor([f_exact(z, Q) for z in z_grid.numpy()],
                            dtype=torch.float64)
    h_target = torch.tensor([h_exact(z, Q) for z in z_grid.numpy()],
                            dtype=torch.float64)

    opt_f = optim.Adam(v_model.f_net.parameters(), lr=lr)
    opt_h = optim.Adam(v_model.h_net.parameters(), lr=lr)

    t0 = time.time()
    for step in range(1, n_steps + 1):
        # f
        opt_f.zero_grad()
        f_pred = v_model.forward_f(z_grid)
        loss_f = ((f_pred - f_target) ** 2).mean()
        loss_f.backward()
        opt_f.step()

        # h
        opt_h.zero_grad()
        h_pred = v_model.forward_h(z_grid)
        loss_h = ((h_pred - h_target) ** 2).mean()
        loss_h.backward()
        opt_h.step()

        if step % 5000 == 0 or step == 1:
            with torch.no_grad():
                fp = v_model.forward_f(z_grid).numpy()
                hp = v_model.forward_h(z_grid).numpy()
            f_err = np.max(np.abs(fp - f_target.numpy()))
            h_err = np.max(np.abs(hp - h_target.numpy()))
            # derivatives
            dz = 1e-4
            z01 = torch.tensor([0.0, dz], dtype=torch.float64)
            with torch.no_grad():
                fv = v_model.forward_f(z01).numpy()
                hv = v_model.forward_h(z01).numpy()
            fp0 = (fv[1] - fv[0]) / dz
            hp0 = (hv[1] - hv[0]) / dz
            print(f"  f/h step {step:6d}: f_max_err={f_err:.6f} "
                  f"h_max_err={h_err:.6f} "
                  f"f'(0)={fp0:.4f} h'(0)={hp0:.4f} "
                  f"({time.time()-t0:.1f}s)")

    print(f"  f/h pretraining done.")


def pretrain_L(l_model, v_model, n_steps=50000, lr=1e-4, n_grid=4000):
    """Train L to minimize area with exact f, h (frozen)."""
    from scipy.interpolate import interp1d

    # Load ODE data for S_EE targets
    ode = np.load(os.path.join(base_dir, 'data', 'ode_benchmark_gr_d3.npz'))
    l_c = float(ode['l_c'][0])
    z_star_grid = ode['z_star_grid']
    l_of_zstar = ode['l_of_zstar']
    A_reg_of_zstar = ode['A_reg_of_zstar']
    l_to_zs = interp1d(l_of_zstar, z_star_grid,
                        kind='cubic', fill_value='extrapolate')
    zs_to_A = interp1d(z_star_grid, A_reg_of_zstar,
                        kind='cubic', fill_value='extrapolate')

    l_min, l_max = 0.15, 0.95 * l_c
    S_interp = interp1d(
        np.linspace(l_min, l_max, 50),
        [float(zs_to_A(float(l_to_zs(lv))))
         for lv in np.linspace(l_min, l_max, 50)],
        kind='cubic', fill_value='extrapolate')

    # Freeze f, h
    for p in v_model.parameters():
        p.requires_grad_(False)

    opt_L = optim.Adam(l_model.parameters(), lr=lr)
    t0 = time.time()

    for step in range(1, n_steps + 1):
        l_val = float(np.random.uniform(l_min, l_max))
        x_np = make_half_grid(l_val, n_grid, x_s_frac=X_S_FRAC)
        A_reg = rt_area_V_gr_d3(l_model, v_model, x_np, l_val)

        opt_L.zero_grad()
        loss = A_reg  # minimize area
        loss.backward()
        opt_L.step()

        if step % 10000 == 0 or step == 1:
            # Check A vs S at a few l values
            l_test = np.linspace(l_min, l_max, 10)
            errs = []
            l_model.eval()
            for lv in l_test:
                x_np = make_half_grid(lv, n_grid, x_s_frac=X_S_FRAC)
                A = rt_area_V_gr_d3(l_model, v_model, x_np, lv).item()
                S = float(S_interp(lv))
                errs.append(abs(A - S))
            l_model.train()
            print(f"  L step {step:6d}: mean|A-S|={np.mean(errs):.6f} "
                  f"max|A-S|={np.max(errs):.6f} "
                  f"({time.time()-t0:.1f}s)")

    # Unfreeze f, h
    for p in v_model.parameters():
        p.requires_grad_(True)

    print(f"  L pretraining done.")


def main():
    print("=" * 60)
    print("Pretraining f, h, L to exact GR solution")
    print("=" * 60)

    v_model = VModelGR_d3(z_h=Z_H, h_h=h_h, hidden=20, depth=2)
    l_model = LModel(hidden=32, depth=2, d=D_BOUNDARY_D3)

    print("\n--- Step 1: Pretrain f and h ---")
    pretrain_f_h(v_model, n_steps=100000, lr=1e-3)

    print("\n--- Step 2: Pretrain L ---")
    pretrain_L(l_model, v_model, n_steps=50000, lr=1e-4)

    # Save checkpoint
    ckpt_path = os.path.join(base_dir, 'data', 'pretrained_exact.pt')
    torch.save({
        'epoch': 0,
        'l_model': l_model.state_dict(),
        'v_model': v_model.state_dict(),
    }, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    # Final diagnostics
    z_pts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    z_t = torch.tensor(z_pts, dtype=torch.float64)
    with torch.no_grad():
        f_pred = v_model.forward_f(z_t).numpy()
        h_pred = v_model.forward_h(z_t).numpy()
    f_ex = [f_exact(z, Q) for z in z_pts]
    h_ex = [h_exact(z, Q) for z in z_pts]

    print("\nFinal f errors:")
    for z, fp, fe in zip(z_pts, f_pred, f_ex):
        print(f"  f({z})={fp:.6f} exact={fe:.6f} err={abs(fp-fe):.6f}")

    print("\nFinal h errors:")
    for z, hp, he in zip(z_pts, h_pred, h_ex):
        print(f"  h({z})={hp:.6f} exact={he:.6f} err={abs(hp-he):.6f}")

    dz = 1e-4
    z01 = torch.tensor([0.0, dz], dtype=torch.float64)
    with torch.no_grad():
        fv = v_model.forward_f(z01).numpy()
        hv = v_model.forward_h(z01).numpy()
    print(f"\nf'(0)={(fv[1]-fv[0])/dz:.6f} h'(0)={(hv[1]-hv[0])/dz:.6f} "
          f"exact={1.5*Q:.6f}")

    print("\nDone. Now run ann_inverse_gr_d3.py with resume from "
          f"{ckpt_path}")


if __name__ == "__main__":
    main()
