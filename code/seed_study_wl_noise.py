"""
Noise-robustness study for the ACTUAL method: three-network reconstruction
from BOTH observables (S_EE + V(L)), with noise on the observable data only.

Per the project rule (never reconstruct from a single observable -- S_EE alone
is degenerate), the noise study uses the joint S_EE + V(L) reconstruction.
Multiplicative Gaussian noise is applied to the two OBSERVABLE arrays only,
with independent streams; coordinates (l, L), exact metric, and networks stay
clean:
    S_noisy = S_clean (1 + sigma xi_S),   V_noisy = V_clean (1 + sigma xi_V).

Design:
  - 5 realizations per sigma in {0.1%, 1%, 5%}: (train_seed, noise_seed) pairs.
  - 500k epochs each (full precision, matching Study C's extended runs) so the
    noise response is read against the ~0.15% converged floor, not a coarse
    short-run floor.
  - sigma=0 baseline: reuse Study C's 4 clean 500k runs already in wl_500k/.
  - Fresh from epoch 1, no resume; jobs run from a scratch cwd.

Outputs: data/seed_study/wl_noise500k/wl_sig*_ts*_ns*.npz
Usage:   caffeinate -i python code/seed_study_wl_noise.py
Status:  data/seed_study/status_wl_noise.json
"""
import os
import sys
import json
import time
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
LOG_DIR = os.path.join(BASE_DIR, 'logs', 'seed_study')
JOB_CWD = os.path.join(LOG_DIR, '.jobcwd')
OUT_BASE = os.path.join(BASE_DIR, 'data', 'seed_study')
STATUS = os.path.join(OUT_BASE, 'status_wl_noise.json')
PY = sys.executable

# 300k is enough to conclude convergence (the both-observable method has no
# flat direction to stall in); 500k only matched Study C's precision. Both
# configurable via env so each sigma-wave can be launched independently.
EPOCHS = int(os.environ.get('SS_EPOCHS', '300000'))
SIGMAS = [float(x) for x in os.environ.get('SS_SIGMAS', '0.01,0.001,0.05').split(',')]
MAX_WORKERS = 5
PAIRS = [(1, 301), (2, 302), (3, 303), (4, 304), (5, 305)]

os.makedirs(JOB_CWD, exist_ok=True)
_lock = threading.Lock()
_state = {}


def set_status(name, **kw):
    with _lock:
        _state.setdefault(name, {}).update(kw)
        with open(STATUS, 'w') as fh:
            json.dump(_state, fh, indent=1, sort_keys=True)


def build_jobs():
    jobs = []
    for sig in SIGMAS:
        for ts, ns in PAIRS:
            jobs.append({
                'name': f"wlnoise{sig:g}_t{ts:02d}",
                'argv': ['--seed', str(ts), '--sigma', str(sig),
                         '--noise-seed', str(ns), '--epochs', str(EPOCHS),
                         '--outdir-suffix', '_noise500k', '--ckpt-every', '50000'],
                'out': os.path.join(OUT_BASE, 'wl_noise500k',
                                    f"wl_sig{sig:.3f}_ts{ts:02d}_ns{ns}.npz"),
            })
    return jobs


def run_job(job):
    name, out = job['name'], job['out']
    if os.path.exists(out):
        set_status(name, status='skipped (output exists)')
        print(f"[skip] {name}", flush=True)
        return
    env = dict(os.environ)
    env.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'})
    argv = [PY, os.path.join(CODE_DIR, 'seed_study_wl.py')] + job['argv']
    t0 = time.time()
    set_status(name, status='running', started=time.strftime('%H:%M:%S'))
    print(f"[start] {name}", flush=True)
    with open(os.path.join(LOG_DIR, f"{name}.log"), 'w') as log:
        rc = subprocess.run(['nice', '-n', '15'] + argv,
                            stdout=log, stderr=subprocess.STDOUT,
                            env=env, cwd=JOB_CWD).returncode
    mins = (time.time() - t0) / 60.0
    ok = (rc == 0) and os.path.exists(out)
    set_status(name, status='done' if ok else f'FAILED (rc={rc})',
               minutes=round(mins, 1))
    print(f"[{'done' if ok else 'FAIL'}] {name}  ({mins:.1f} min)", flush=True)


def main():
    jobs = build_jobs()
    print(f"{len(jobs)} WL noise jobs (both observables), {MAX_WORKERS} workers, "
          f"{EPOCHS} epochs each", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(ex.map(run_job, jobs))
    print(f"ALL DONE in {(time.time()-t0)/3600:.2f} h", flush=True)
    n_fail = sum(1 for v in _state.values()
                 if str(v.get('status', '')).startswith('FAILED'))
    print(f"failures: {n_fail}", flush=True)


if __name__ == "__main__":
    main()
