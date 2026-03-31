"""
Test script for cipher implementations.
Verifies correctness against known test vectors from the original papers.
"""

import numpy as np
import sys

def test_speck32():
    """Test SPECK32 implementation with known test vectors."""
    from ciphers.speck32 import Speck32
    
    cipher = Speck32()
    
    # Test vector from SPECK paper
    # Key: 0x1918 1110 0908 0100
    key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    plaintext = np.array([0x6574694c], dtype=np.uint32)  # "Lite"
    
    # Expected ciphertext after 22 rounds: 0xa868 42f2
    ciphertext = cipher.encrypt(plaintext, 22, key)
    expected = 0xa86842f2
    
    print(f"  Plaintext:  0x{plaintext[0]:08x}")
    print(f"  Ciphertext: 0x{ciphertext[0]:08x}")
    print(f"  Expected:   0x{expected:08x}")
    
    return ciphertext[0] == expected


def test_simon32():
    """Test SIMON32 implementation with known test vectors."""
    from ciphers.simon32 import Simon32
    
    cipher = Simon32()
    
    # Test vector from SIMON paper
    # Key: 0x1918 1110 0908 0100
    key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    plaintext = np.array([0x65656877], dtype=np.uint32)
    
    # Expected ciphertext after 32 rounds: 0xc69b e9bb
    ciphertext = cipher.encrypt(plaintext, 32, key)
    expected = 0xc69be9bb
    
    print(f"  Plaintext:  0x{plaintext[0]:08x}")
    print(f"  Ciphertext: 0x{ciphertext[0]:08x}")
    print(f"  Expected:   0x{expected:08x}")
    
    return ciphertext[0] == expected


def test_present():
    """Test PRESENT implementation with known test vectors."""
    from ciphers.present import Present
    
    cipher = Present()
    
    # Test vector from PRESENT paper
    # Key: 0x00000000000000000000 (80 bits)
    key = np.array([0x0000, 0x0000000000000000], dtype=np.uint64)
    plaintext = np.array([0x0000000000000000], dtype=np.uint64)
    
    # Expected ciphertext after 31 rounds: 0x5579C1387B228445
    ciphertext = cipher.encrypt(plaintext, 31, key)
    expected = 0x5579C1387B228445
    
    print(f"  Plaintext:  0x{plaintext[0]:016x}")
    print(f"  Ciphertext: 0x{ciphertext[0]:016x}")
    print(f"  Expected:   0x{expected:016x}")
    
    return ciphertext[0] == expected


def test_random_permutation():
    """Test RandomPermutation for basic functionality."""
    from ciphers.random_permutation import RandomPermutation
    
    rp32 = RandomPermutation(block_size=32)
    rp64 = RandomPermutation(block_size=64)
    
    # Test 32-bit
    plaintexts = np.array([0x12345678, 0xDEADBEEF], dtype=np.uint32)
    c32 = rp32.encrypt(plaintexts)
    print(f"  32-bit: Generated {len(c32)} random outputs")
    
    # Test 64-bit
    plaintexts64 = np.array([0x123456789ABCDEF0], dtype=np.uint64)
    c64 = rp64.encrypt(plaintexts64)
    print(f"  64-bit: Generated {len(c64)} random outputs")
    
    # Test random pairs
    c, c_prime = rp32.generate_random_pairs(100)
    print(f"  Random pairs: Generated {len(c)} pairs")
    
    # Check they're independent (different values)
    diff_count = np.sum(c != c_prime)
    print(f"  Independence check: {diff_count}/100 pairs are different")
    
    return diff_count > 90  # Allow some collisions but mostly different


def test_batch_operations():
    """Test batch encryption works correctly."""
    from ciphers.speck32 import Speck32
    from ciphers.simon32 import Simon32
    
    print("  Testing batch encryption...")
    
    # SPECK32 batch test
    speck = Speck32()
    key = speck.random_key()
    plaintexts = speck.random_plaintexts(1000)
    ciphertexts = speck.encrypt(plaintexts, 22, key)
    
    # Verify single encrypt matches batch
    single_ct = speck.encrypt(plaintexts[:1], 22, key)
    batch_match = ciphertexts[0] == single_ct[0]
    print(f"  SPECK32 batch: {len(ciphertexts)} encrypted, single/batch match: {batch_match}")
    
    # SIMON32 batch test
    simon = Simon32()
    key = simon.random_key()
    plaintexts = simon.random_plaintexts(1000)
    ciphertexts = simon.encrypt(plaintexts, 32, key)
    
    single_ct = simon.encrypt(plaintexts[:1], 32, key)
    batch_match2 = ciphertexts[0] == single_ct[0]
    print(f"  SIMON32 batch: {len(ciphertexts)} encrypted, single/batch match: {batch_match2}")
    
    return batch_match and batch_match2


def test_encrypt_with_trace():
    """Test encrypt_with_trace returns correct intermediate states."""
    from ciphers.speck32 import Speck32
    from ciphers.simon32 import Simon32
    
    print("  Testing encrypt_with_trace...")
    
    # SPECK32
    speck = Speck32()
    key = speck.random_key()
    plaintext = speck.random_plaintexts(10)
    
    final, states = speck.encrypt_with_trace(plaintext, 5, key)
    print(f"  SPECK32: {len(states)} intermediate states for 5 rounds")
    
    # Check final matches last state
    final_match = np.all(final == states[-1])
    print(f"  SPECK32: Final matches last state: {final_match}")
    
    # SIMON32
    simon = Simon32()
    key = simon.random_key()
    plaintext = simon.random_plaintexts(10)
    
    final, states = simon.encrypt_with_trace(plaintext, 5, key)
    print(f"  SIMON32: {len(states)} intermediate states for 5 rounds")
    
    final_match2 = np.all(final == states[-1])
    print(f"  SIMON32: Final matches last state: {final_match2}")
    
    return len(states) == 5 and final_match and len(states) == 5 and final_match2


def test_differential():
    """Test that differential pairs can be created correctly."""
    from ciphers.speck32 import Speck32
    
    print("  Testing differential pair creation...")
    
    speck = Speck32()
    key = speck.random_key()
    delta_p = speck.get_default_delta_p()
    
    # Create plaintext and its differential pair
    P = speck.random_plaintexts(100)
    P_prime = speck.apply_difference(P, delta_p)
    
    # Verify XOR difference is correct
    actual_diff = P ^ P_prime
    diff_correct = np.all(actual_diff == delta_p)
    print(f"  Input difference 0x{delta_p:08x}: correct = {diff_correct}")
    
    # Encrypt both
    C = speck.encrypt(P, 5, key)
    C_prime = speck.encrypt(P_prime, 5, key)
    
    # Output difference
    delta_c = C ^ C_prime
    unique_diffs = len(np.unique(delta_c))
    print(f"  Output differences: {unique_diffs} unique values from 100 pairs")
    
    return diff_correct


def main():
    print("=" * 60)
    print("CIPHER IMPLEMENTATION TESTS")
    print("=" * 60)
    
    results = {}
    
    print("\n[1/6] SPECK32/64 Test Vector:")
    results['SPECK32'] = test_speck32()
    
    print("\n[2/6] SIMON32/64 Test Vector:")
    results['SIMON32'] = test_simon32()
    
    print("\n[3/6] PRESENT-80 Test Vector:")
    results['PRESENT'] = test_present()
    
    print("\n[4/6] Random Permutation:")
    results['RandomPerm'] = test_random_permutation()
    
    print("\n[5/6] Batch Operations:")
    results['Batch'] = test_batch_operations()
    
    print("\n[6/6] Encrypt with Trace:")
    results['Trace'] = test_encrypt_with_trace()
    
    print("\n[7/7] Differential Pairs:")
    results['Differential'] = test_differential()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}: {status}")
        if not passed:
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
