import numpy as np

from gen_bit import gen_bit
from qam16_mod import qam16_mod
from sfbc_encode import sfbc_encode


N_FFT = 64
N_BITS = N_FFT * 4


def test_flow(seed=2026):
    # 1. Sinh bit
    bits = gen_bit(N_BITS, rng=seed)

    # 2. 16-QAM modulation
    s = qam16_mod(bits)

    # 3. SFBC encoding
    X = sfbc_encode(s)

    # Kiểm tra kích thước cơ bản của toàn bộ flow
    assert bits.shape == (N_BITS,)
    assert s.shape == (N_FFT,)
    assert X.shape == (2, N_FFT // 2)

    # Kiểm tra không có NaN / Inf
    assert np.all(np.isfinite(s))
    assert np.all(np.isfinite(X))

    # Kiểm tra công suất
    power = np.mean(np.sum(np.abs(X) ** 2, axis=0))
    assert np.isclose(power, 1.0, atol=0.02)

    print("[OK] Flow: gen_bit -> qam16_mod -> sfbc_encode")

    print(f"\nFlow:")
    print(f"  {bits.size} bit")
    print(f"  -> {s.size} ký hiệu 16-QAM")
    print(f"  -> SFBC grid {X.shape}")

    print("\n8 bit đầu:")
    print(bits[:8])

    print("\ns1, s2:")
    print(np.round(s[:2] * np.sqrt(10), 3),
          "(đơn vị 1/sqrt10)")

    print("\nSFBC:")
    print("Anten 1:", np.round(X[0, :2] * np.sqrt(20), 3),
          "(đơn vị 1/sqrt20)")
    print("Anten 2:", np.round(X[1, :2] * np.sqrt(20), 3),
          "(đơn vị 1/sqrt20)")

    print(f"\nCông suất tổng/bin = {power:.4f}")


if __name__ == "__main__":
    test_flow()
    print("\nFlow test đạt.")