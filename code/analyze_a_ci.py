"""
Confidence-interval analysis of the recovered boundary derivative
a = f'(0) = h'(0) vs noise level sigma, for the both-observable
(S_EE + V(L)) reconstruction, all evaluated at the SAME epoch count so the
noise effect is isolated (the mid-training overshoot is common to all levels).

For each sigma the 5 (or 4) realizations are treated as a sample:
  mean, sample std, standard error, 95% t-CI (dof = n-1),
  bias |mean - 1.5|, and whether the exact value 1.5 lies inside the CI.
Expectation: the spread / CI width of a grows with sigma at fixed epochs.

Sources of a at the target epoch:
  sigma=0  -> Study C clean runs (wl_500k/wl_seed*.npz), a_hist at target
  sigma>0  -> noise runs (wl_noise500k/wl_sig*_ts*.npz), a_final (300k eval)

Usage: python code/analyze_a_ci.py [target_epoch]   (default 300000)
"""
import os
import sys
import glob
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SS = os.path.join(BASE, 'data', 'seed_study')
A_EXACT = 1.5
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 300000


def a_clean_at(target):
    """a values for the sigma=0 runs, read from a_hist at the target epoch."""
    vals = []
    for f in sorted(glob.glob(os.path.join(SS, 'wl_500k', 'wl_seed*.npz'))):
        d = np.load(f)
        if 'a_hist' in d.files and len(d['a_hist']):
            ep = d['a_hist_ep']
            i = np.argmin(np.abs(ep - target))
            vals.append(float(d['a_hist'][i]))
    return np.array(vals)


def a_noise(sigma):
    """a_final for the noise runs at the given sigma (evaluated at 300k)."""
    vals = []
    for f in sorted(glob.glob(os.path.join(
            SS, 'wl_noise500k', f'wl_sig{sigma:.3f}_ts*.npz'))):
        d = np.load(f)
        vals.append(float(d['a_final']))
    return np.array(vals)


def report(label, a):
    n = len(a)
    if n == 0:
        print(f"{label:>10s} | (no data yet)")
        return
    mean, sd = a.mean(), a.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else float('nan')
    half = tcrit * se
    lo, hi = mean - half, mean + half
    inside = lo <= A_EXACT <= hi
    bias = abs(mean - A_EXACT)
    print(f"{label:>10s} | n={n} | mean={mean:.4f}  sd={sd:.4f}  SE={se:.4f} "
          f"| 95% CI=[{lo:.4f}, {hi:.4f}]  half-width={half:.4f} "
          f"| bias={bias:.4f} | 1.5 inside CI? {'YES' if inside else 'NO'}")
    return dict(sigma=label, n=n, mean=mean, sd=sd, se=se, ci=(lo, hi),
                inside=inside, bias=bias, vals=a)


def main():
    print(f"a = f'(0) = h'(0) confidence-interval analysis @ epoch {TARGET} "
          f"(exact a = {A_EXACT})\n")
    print("  sigma=0 (clean):", np.round(a_clean_at(TARGET), 4))
    for s in (0.01, 0.05, 0.10):
        v = a_noise(s)
        if len(v):
            print(f"  sigma={s*100:g}%:", np.round(v, 4))
    print()
    report('sigma=0', a_clean_at(TARGET))
    report('sigma=1%', a_noise(0.01))
    report('sigma=5%', a_noise(0.05))
    report('sigma=10%', a_noise(0.10))


if __name__ == "__main__":
    main()
