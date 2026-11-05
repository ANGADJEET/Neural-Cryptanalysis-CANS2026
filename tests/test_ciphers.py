
import numpy as np
import sys
from pathlib import Path

# So the file also runs directly as a script, not only under pytest.
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_speck32():
    from ciphers.speck32 import Speck32

    cipher = Speck32()

    key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    plaintext = np.array([0x6574694c], dtype=np.uint32)

    ciphertext = cipher.encrypt(plaintext, 22, key)
    expected = 0xa86842f2

    print(f"  Plaintext:  0x{plaintext[0]:08x}")
    print(f"  Ciphertext: 0x{ciphertext[0]:08x}")
    print(f"  Expected:   0x{expected:08x}")

    assert ciphertext[0] == expected, \
        f"SPECK32/64 test vector: got 0x{ciphertext[0]:08x}, expected 0x{expected:08x}"


def test_simon32():
    from ciphers.simon32 import Simon32

    cipher = Simon32()

    key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    plaintext = np.array([0x65656877], dtype=np.uint32)

    ciphertext = cipher.encrypt(plaintext, 32, key)
    expected = 0xc69be9bb

    print(f"  Plaintext:  0x{plaintext[0]:08x}")
    print(f"  Ciphertext: 0x{ciphertext[0]:08x}")
    print(f"  Expected:   0x{expected:08x}")

    assert ciphertext[0] == expected, \
        f"SIMON32/64 test vector: got 0x{ciphertext[0]:08x}, expected 0x{expected:08x}"


def test_simon32_z_sequence():
    from ciphers.simon32 import Simon32

    # z0 from Beaulieu et al., period 62.
    official = "11111010001001010110000111001101111101000100101011000011100110"

    assert len(Simon32.Z_SEQ) == 62, f"z0 has period 62, found {len(Simon32.Z_SEQ)}"
    assert Simon32.Z_SEQ == [int(b) for b in official], "Z_SEQ does not match z0"


def test_present():
    from ciphers.present import Present

    cipher = Present()

    # All four vectors from the PRESENT paper, as (plaintext, key, ciphertext).
    # The key is split as (high 16 bits, low 64 bits).
    vectors = [
        (0x0000000000000000, (0x0000, 0x0000000000000000), 0x5579C1387B228445),
        (0x0000000000000000, (0xFFFF, 0xFFFFFFFFFFFFFFFF), 0xE72C46C0F5945049),
        (0xFFFFFFFFFFFFFFFF, (0x0000, 0x0000000000000000), 0xA112FFC72F68417B),
        (0xFFFFFFFFFFFFFFFF, (0xFFFF, 0xFFFFFFFFFFFFFFFF), 0x3333DCD3213210D2),
    ]

    for pt, k, expected in vectors:
        key = np.array(list(k), dtype=np.uint64)
        plaintext = np.array([pt], dtype=np.uint64)

        ciphertext = cipher.encrypt(plaintext, 31, key)

        print(f"  Plaintext:  0x{plaintext[0]:016x}")
        print(f"  Ciphertext: 0x{ciphertext[0]:016x}")
        print(f"  Expected:   0x{expected:016x}")

        assert ciphertext[0] == expected, \
            f"PRESENT-80 vector pt=0x{pt:016x} key=0x{k[0]:04x}{k[1]:016x}: " \
            f"got 0x{ciphertext[0]:016x}, expected 0x{expected:016x}"


def test_random_permutation():
    from ciphers.random_permutation import RandomPermutation

    rp32 = RandomPermutation(block_size=32)
    rp64 = RandomPermutation(block_size=64)

    plaintexts = np.array([0x12345678, 0xDEADBEEF], dtype=np.uint32)
    c32 = rp32.encrypt(plaintexts)
    print(f"  32-bit: Generated {len(c32)} random outputs")
    assert len(c32) == len(plaintexts)

    plaintexts64 = np.array([0x123456789ABCDEF0], dtype=np.uint64)
    c64 = rp64.encrypt(plaintexts64)
    print(f"  64-bit: Generated {len(c64)} random outputs")
    assert len(c64) == len(plaintexts64)

    c, c_prime = rp32.generate_random_pairs(100)
    print(f"  Random pairs: Generated {len(c)} pairs")
    assert len(c) == 100 and len(c_prime) == 100

    diff_count = np.sum(c != c_prime)
    print(f"  Independence check: {diff_count}/100 pairs are different")

    # Two independent 32-bit draws collide with probability 2^-32, so any
    # sizeable number of equal pairs means the outputs are not independent.
    assert diff_count > 90, f"only {diff_count}/100 pairs differ"


def test_batch_operations():
    from ciphers.speck32 import Speck32
    from ciphers.simon32 import Simon32

    print("  Testing batch encryption...")

    speck = Speck32()
    key = speck.random_key()
    plaintexts = speck.random_plaintexts(1000)
    ciphertexts = speck.encrypt(plaintexts, 22, key)

    single_ct = speck.encrypt(plaintexts[:1], 22, key)
    batch_match = ciphertexts[0] == single_ct[0]
    print(f"  SPECK32 batch: {len(ciphertexts)} encrypted, single/batch match: {batch_match}")

    assert len(ciphertexts) == 1000
    assert batch_match, "SPECK32 batch and single-sample encryption disagree"

    simon = Simon32()
    key = simon.random_key()
    plaintexts = simon.random_plaintexts(1000)
    ciphertexts = simon.encrypt(plaintexts, 32, key)

    single_ct = simon.encrypt(plaintexts[:1], 32, key)
    batch_match2 = ciphertexts[0] == single_ct[0]
    print(f"  SIMON32 batch: {len(ciphertexts)} encrypted, single/batch match: {batch_match2}")

    assert len(ciphertexts) == 1000
    assert batch_match2, "SIMON32 batch and single-sample encryption disagree"


def test_encrypt_with_trace():
    from ciphers.speck32 import Speck32
    from ciphers.simon32 import Simon32

    print("  Testing encrypt_with_trace...")

    speck = Speck32()
    key = speck.random_key()
    plaintext = speck.random_plaintexts(10)

    final, states = speck.encrypt_with_trace(plaintext, 5, key)
    print(f"  SPECK32: {len(states)} intermediate states for 5 rounds")

    final_match = np.all(final == states[-1])
    print(f"  SPECK32: Final matches last state: {final_match}")

    assert len(states) == 5, f"SPECK32: expected 5 states, got {len(states)}"
    assert final_match, "SPECK32: final output differs from the last traced state"

    # The trace must agree with a plain encryption at every round.
    for r in range(1, 6):
        assert np.all(speck.encrypt(plaintext, r, key) == states[r-1]), \
            f"SPECK32: trace disagrees with encrypt at round {r}"

    simon = Simon32()
    key = simon.random_key()
    plaintext = simon.random_plaintexts(10)

    final, states = simon.encrypt_with_trace(plaintext, 5, key)
    print(f"  SIMON32: {len(states)} intermediate states for 5 rounds")

    final_match2 = np.all(final == states[-1])
    print(f"  SIMON32: Final matches last state: {final_match2}")

    assert len(states) == 5, f"SIMON32: expected 5 states, got {len(states)}"
    assert final_match2, "SIMON32: final output differs from the last traced state"

    for r in range(1, 6):
        assert np.all(simon.encrypt(plaintext, r, key) == states[r-1]), \
            f"SIMON32: trace disagrees with encrypt at round {r}"


def test_differential():
    from ciphers.speck32 import Speck32

    print("  Testing differential pair creation...")

    speck = Speck32()
    key = speck.random_key()
    delta_p = speck.get_default_delta_p()

    P = speck.random_plaintexts(100)
    P_prime = speck.apply_difference(P, delta_p)

    actual_diff = P ^ P_prime
    diff_correct = np.all(actual_diff == delta_p)
    print(f"  Input difference 0x{delta_p:08x}: correct = {diff_correct}")

    assert diff_correct, "apply_difference did not produce the requested input difference"

    C = speck.encrypt(P, 5, key)
    C_prime = speck.encrypt(P_prime, 5, key)

    delta_c = C ^ C_prime
    unique_diffs = len(np.unique(delta_c))
    print(f"  Output differences: {unique_diffs} unique values from 100 pairs")

    # A fixed nonzero input difference cannot vanish, and must not survive
    # 5 rounds as a single fixed output difference.
    assert np.all(delta_c != 0), "some pairs collided after 5 rounds"
    assert unique_diffs > 1, "output difference is constant across 100 pairs"


def main():
    print("=" * 60)
    print("CIPHER IMPLEMENTATION TESTS")
    print("=" * 60)

    checks = [
        ('SPECK32', test_speck32),
        ('SIMON32', test_simon32),
        ('SIMON32 z0', test_simon32_z_sequence),
        ('PRESENT', test_present),
        ('RandomPerm', test_random_permutation),
        ('Batch', test_batch_operations),
        ('Trace', test_encrypt_with_trace),
        ('Differential', test_differential),
    ]

    results = {}

    for i, (name, fn) in enumerate(checks, start=1):
        print(f"\n[{i}/{len(checks)}] {name}:")
        try:
            fn()
            results[name] = None
        except AssertionError as e:
            results[name] = str(e)
            print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, failure in results.items():
        status = "PASS" if failure is None else "FAIL"
        print(f"  [{status}] {name}")
        if failure is not None:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
