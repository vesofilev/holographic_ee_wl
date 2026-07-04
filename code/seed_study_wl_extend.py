"""
Extend selected Study C (three-network WL) seeds from 50k -> 500k epochs, to
reach the paper's precision (0.14%/0.17%) and show it is seed-independent.

Resumes each seed from its 50k checkpoint. The FIRST resume of each seed is
model-only (the 50k checkpoints predate optimizer-saving -> one unavoidable
Adam reset, verified benign because the 50k state is already converged). From
then on, seed_study_wl.py writes full model+optimizer+RNG checkpoints every
50k into wl_500k/, so any interrupt-restart resumes faithfully.

Output: data/seed_study/wl_500k/wl_seedNN.npz  (originals in wl_50k/ untouched)
Usage:  caffeinate -i python code/seed_study_wl_extend.py
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
JOB_CWD = os.path.join(LOG_DIR, '.jobcwd')   # run here so import-time log
                                              # truncation can't touch repo logs
OUT_BASE = os.path.join(BASE_DIR, 'data', 'seed_study')
STATUS = os.path.join(OUT_BASE, 'status_wl_extend.json')
PY = sys.executable

SEEDS = [5, 7, 3, 2]        # span the a-range (1.492 .. 1.508)
TARGET_EPOCHS = 500000
MAX_WORKERS = 4

os.makedirs(JOB_CWD, exist_ok=True)
_lock = threading.Lock()
_state = {}


def set_status(name, **kw):
    with _lock:
        _state.setdefault(name, {}).update(kw)
        with open(STATUS, 'w') as fh:
            json.dump(_state, fh, indent=1, sort_keys=True)


def run_job(seed):
    name = f"wl_s{seed:02d}"
    out_npz = os.path.join(OUT_BASE, 'wl_500k', f"wl_seed{seed:02d}.npz")
    if os.path.exists(out_npz):
        set_status(name, status='skipped (output exists)')
        print(f"[skip] {name}", flush=True)
        return

    # Faithful restart: if a wl_500k checkpoint already exists (interrupted
    # run), resume from it (has optimizer+RNG); else start from the 50k one.
    ckpt_500k = os.path.join(OUT_BASE, 'wl_500k', f"wl_seed{seed:02d}_models.pt")
    ckpt_50k = os.path.join(OUT_BASE, 'wl_50k', f"wl_seed{seed:02d}_models.pt")
    resume_from = ckpt_500k if os.path.exists(ckpt_500k) else ckpt_50k
    if not os.path.exists(resume_from):
        set_status(name, status='FAILED (no 50k checkpoint)')
        print(f"[FAIL] {name}: missing {resume_from}", flush=True)
        return

    env = dict(os.environ)
    env.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'})
    argv = [PY, os.path.join(CODE_DIR, 'seed_study_wl.py'),
            '--seed', str(seed), '--epochs', str(TARGET_EPOCHS),
            '--outdir-suffix', '_500k', '--ckpt-every', '50000',
            '--resume-from', resume_from]

    logpath = os.path.join(LOG_DIR, f"wl_s{seed:02d}_extend.log")
    t0 = time.time()
    set_status(name, status='running', started=time.strftime('%H:%M:%S'),
               resumed_from=os.path.basename(os.path.dirname(resume_from)))
    print(f"[start] {name}  (from {resume_from.split('seed_study/')[-1]})", flush=True)
    with open(logpath, 'w') as log:
        rc = subprocess.run(['nice', '-n', '15'] + argv,
                            stdout=log, stderr=subprocess.STDOUT,
                            env=env, cwd=JOB_CWD).returncode
    mins = (time.time() - t0) / 60.0
    ok = (rc == 0) and os.path.exists(out_npz)
    set_status(name, status='done' if ok else f'FAILED (rc={rc})',
               minutes=round(mins, 1))
    print(f"[{'done' if ok else 'FAIL'}] {name}  ({mins:.1f} min)", flush=True)


def main():
    print(f"Extending {len(SEEDS)} WL seeds {SEEDS} to {TARGET_EPOCHS} epochs, "
          f"{MAX_WORKERS} workers", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(ex.map(run_job, SEEDS))
    print(f"ALL DONE in {(time.time()-t0)/3600:.2f} h", flush=True)
    n_fail = sum(1 for v in _state.values()
                 if str(v.get('status', '')).startswith('FAILED'))
    print(f"failures: {n_fail}", flush=True)


if __name__ == "__main__":
    main()
