"""
Fresh noise-robustness ensemble for the h=1 inverse problem (revised paper
noise section; referee point 3: "Multiple realizations should be examined
before quantitative conclusions are drawn").

Design (replaces the paper's single-realization-per-level table):
  - 6 realizations per noise level sigma in {0.1%, 1%, 5%}: five new
    (train_seed, noise_seed) pairs + the paper's original pair (42, 123).
  - sigma = 0 baseline ensemble (seeds 4, 5, 42; seeds 1-3 already done in
    h1_200k) calibrating the method's intrinsic optimizer variance: the
    noise response is the rise of the error above this baseline.
  - ALL runs fresh from epoch 1 (no resume -- resuming from mid-plateau
    model-only checkpoints was the artifact that poisoned the first attempt),
    fixed seeding (seed before model construction), 200k epochs (convergence
    checked via the data-fit residual saved per run).
  - Jobs run from a scratch cwd so import-time FileHandler(mode='w') side
    effects cannot truncate historical logs in the repo root.

Outputs: data/seed_study/noise_fresh200k/*.npz and data/seed_study/h1_200k/
Usage:   caffeinate -i python code/seed_study_noise_fresh.py
Status:  data/seed_study/status_noise_fresh.json
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
STATUS = os.path.join(OUT_BASE, 'status_noise_fresh.json')
PY = sys.executable

EPOCHS = 200000
MAX_WORKERS = 6
PAIRS = [(1, 201), (2, 202), (3, 203), (4, 204), (5, 205), (42, 123)]
SIGMAS = [0.01, 0.05, 0.001]          # problematic 1% level first
CLEAN_SEEDS = [4, 5, 42]              # 1-3 already in h1_200k

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
                'name': f"noise{sig:g}_t{ts:02d}",
                'argv': ['--seed', str(ts), '--sigma', str(sig),
                         '--noise-seed', str(ns), '--epochs', str(EPOCHS),
                         '--outdir-suffix', '_fresh200k'],
                'out': os.path.join(OUT_BASE, 'noise_fresh200k',
                                    f"noise_sig{sig:.3f}_ts{ts:02d}_ns{ns}.npz"),
            })
    for s in CLEAN_SEEDS:
        jobs.append({
            'name': f"h1_s{s:02d}",
            'argv': ['--seed', str(s), '--epochs', str(EPOCHS),
                     '--outdir-suffix', '_200k'],
            'out': os.path.join(OUT_BASE, 'h1_200k', f"h1_seed{s:02d}.npz"),
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
    argv = [PY, os.path.join(CODE_DIR, 'seed_study_h1.py')] + job['argv']
    t0 = time.time()
    set_status(name, status='running', started=time.strftime('%H:%M:%S'))
    print(f"[start] {name}", flush=True)
    with open(os.path.join(LOG_DIR, f"{name}_fresh.log"), 'w') as log:
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
    print(f"{len(jobs)} fresh noise-study jobs, {MAX_WORKERS} workers, "
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
