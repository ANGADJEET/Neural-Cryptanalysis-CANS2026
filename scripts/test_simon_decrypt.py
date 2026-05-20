#!/usr/bin/env python3
"""Test SIMON32 decrypt_one_round correctness."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from ciphers import get_cipher
from experiments.exp12_key_recovery import decrypt_one_round

cipher = get_cipher("simon32")
key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)

for n_rounds in [4, 5, 6, 7]:
    rk = cipher._expand_key(key, n_rounds)
    P = np.array([0x65656877, 0x12345678, 0xABCD1234], dtype=np.uint32)
    C = cipher.encrypt(P, n_rounds, key)
    C_partial = decrypt_one_round("simon32", C, int(rk[-1]))
    C_reduced = cipher.encrypt(P, n_rounds - 1, key)
    match = np.all(C_partial == C_reduced)
    print(f"  {n_rounds}r: subkey=0x{rk[-1]:04x}, match={match}")
    if not match:
        print(f"    C={[hex(int(c)) for c in C]}")
        print(f"    partial={[hex(int(c)) for c in C_partial]}")
        print(f"    reduced={[hex(int(c)) for c in C_reduced]}")

print("\nDone")
