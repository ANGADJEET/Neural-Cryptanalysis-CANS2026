#!/usr/bin/env python3
"""Run multi-bit classical for PRESENT only (remaining from exp29)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import importlib.util

spec = importlib.util.spec_from_file_location("statistics", 
    str(Path(__file__).parent.parent / "data" / "statistics.py"))
stats_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats_mod)
compute_classical_multibit_accuracy = stats_mod.compute_classical_multibit_accuracy

from ciphers import get_cipher

# Collect ALL results (hardcode speck/simon from completed run, run present fresh)
results = {
    "speck32": {
        "3": {"1": {"accuracy": 0.7510}, "2": {"accuracy": 0.7511}, "3": {"accuracy": 0.7516}},
        "4": {"1": {"accuracy": 0.7328}, "2": {"accuracy": 0.7458}, "3": {"accuracy": 0.7309}},
        "5": {"1": {"accuracy": 0.6922}, "2": {"accuracy": 0.7222}, "3": {"accuracy": 0.6935}},
        "6": {"1": {"accuracy": 0.5854}, "2": {"accuracy": 0.6307}, "3": {"accuracy": 0.5874}},
        "7": {"1": {"accuracy": 0.5030}, "2": {"accuracy": 0.5210}, "3": {"accuracy": 0.5082}},
    },
    "simon32": {
        "4": {"1": {"accuracy": 0.7515}, "2": {"accuracy": 0.7520}, "3": {"accuracy": 0.7525}},
        "5": {"1": {"accuracy": 0.7513}, "2": {"accuracy": 0.7515}, "3": {"accuracy": 0.7523}},
        "6": {"1": {"accuracy": 0.7505}, "2": {"accuracy": 0.7506}, "3": {"accuracy": 0.7499}},
        "7": {"1": {"accuracy": 0.6324}, "2": {"accuracy": 0.6052}, "3": {"accuracy": 0.6054}},
        "8": {"1": {"accuracy": 0.5704}, "2": {"accuracy": 0.5487}, "3": {"accuracy": 0.5501}},
    },
}

cipher = get_cipher('present')
delta_p = cipher.get_default_delta_p()
results["present"] = {}
for r in [2, 3, 4, 5, 6]:
    np.random.seed(42)
    print(f"present {r}r: ", end='', flush=True)
    res = compute_classical_multibit_accuracy(
        cipher, delta_p, r,
        n_samples=200000, n_keys=5, max_k=3
    )
    results["present"][str(r)] = {}
    for k in [1, 2, 3]:
        results["present"][str(r)][str(k)] = {"accuracy": res[k]['accuracy']}
        print(f"k={k}: {res[k]['accuracy']:.4f}  ", end='')
    print()

out = Path('results/e29_multibit_classical')
out.mkdir(parents=True, exist_ok=True)
with open(out / 'multibit_classical.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved to {out}")
