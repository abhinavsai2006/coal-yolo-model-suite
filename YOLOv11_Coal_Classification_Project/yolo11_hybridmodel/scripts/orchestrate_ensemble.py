"""
Orchestrate training of multiple seeds and evaluate an ensemble to reach 92%+.
This avoids PowerShell parsing issues by using a Python controller.

Usage:
  E:\Yolo\yolov9_gpu\python.exe scripts\orchestrate_ensemble.py \
      --seeds 456 789 2024 --epochs 40 --lr 0.0001 --patience 15 \
      --python-exe "E:\\Yolo\\yolov9_gpu\\python.exe"
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DEFAULT = str(Path('E:/Yolo/yolov9_gpu/python.exe'))


def run(cmd: list[str]) -> int:
    print("\n>>>", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def find_weights() -> list[str]:
    weights = []
    # existing
    for p in [
        ROOT / 'runs' / 'ensemble_fast_seed42' / 'weights' / 'best.pt',
        ROOT / 'runs' / 'ensemble_fast_seed123' / 'weights' / 'best.pt',
    ]:
        if p.exists():
            weights.append(str(p))
    # new
    for d in (ROOT / 'runs').glob('ensemble_seed*'):
        best = d / 'weights' / 'best.pt'
        if best.exists():
            weights.append(str(best))
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[456, 789, 2024])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--python-exe', type=str, default=PYTHON_DEFAULT)
    args = ap.parse_args()

    py = args.python_exe
    if not Path(py).exists():
        print(f"ERROR: Python not found at {py}")
        return 1

    # Quick CUDA check
    code = run([py, '-c', 'import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)'])
    if code != 0:
        print('ERROR: CUDA not available. Please enable the GPU environment.')
        return 1

    print("\n================ TRAINING NEW MODELS ================")
    for i, seed in enumerate(args.seeds, 1):
        print(f"\n---> Training model {i}/{len(args.seeds)} with seed {seed}")
        cmd = [
            py,
            str(ROOT / 'training' / 'train_ensemble_strategy.py'),
            '--seed', str(seed),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--patience', str(args.patience),
        ]
        start = time.time()
        rc = run(cmd)
        if rc != 0:
            print(f"ERROR: Training failed for seed {seed}")
            return rc
        print(f"Done seed {seed} in {time.time()-start:.1f}s")

    print("\n================ COLLECTING WEIGHTS ================")
    weights = find_weights()
    for w in weights:
        print("  ", w)
    if len(weights) < 3:
        print(f"WARNING: Only {len(weights)} models found; continuing anyway.")

    print("\n================ ENSEMBLE EVALUATION ================")
    out_dir = 'evaluation_final_ensemble'
    eval_cmd = [
        py, str(ROOT / 'inference' / 'evaluate_ensemble_on_test.py'),
        '--weights', *weights,
        '--output', out_dir,
    ]
    rc = run(eval_cmd)
    if rc != 0:
        print('ERROR: Ensemble evaluation failed')
        return rc

    print("\nAll done. See", ROOT / out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
