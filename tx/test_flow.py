import itertools
import numpy as np

from gen_bit import gen_bit
from qam16_mod import qam16_mod
from sfbc_encode import sfbc_encode

N_FFT = 64
N_BITS = N_FFT * 4        


def test_gen_bit():
    b1, b2 = gen_bit(N_BITS, rng=123), gen_bit(N_BITS, rng=123)
    assert b1.shape == (N_BITS,) and set(np.unique(b1)) <= {0, 1}
    assert np.array_equal(b1, b2), "cùng seed phải cho cùng bit"
    assert not np.array_equal(b1, gen_bit(N_BITS, rng=124))
    big = gen_bit(400_000, rng=1)
    assert abs(big.mean() - 0.5) < 0.01
    for bad in (0, 6, -4):
        try:
            gen_bit(bad)
            raise AssertionError("phải báo lỗi")
        except ValueError:
            pass
    print("[OK] gen_bit")


def test_qam16():
    pts = {b: qam16_mod(np.array(b))[0] for b in itertools.product([0, 1], repeat=4)}
    vals = np.array(list(pts.values()))
    assert len(set(np.round(vals, 9))) == 16, "phải đủ 16 điểm khác nhau"
    assert np.isclose(np.mean(np.abs(vals) ** 2), 1.0), "Es phải bằng 1"
    assert np.allclose(qam16_mod([0, 1, 1, 0]), (-1 + 3j) / np.sqrt(10))
    assert np.allclose(qam16_mod([1, 0, 0, 1]), (3 - 1j) / np.sqrt(10))
    # Gray: hai điểm cách nhau d_min chỉ khác đúng 1 bit
    d_min = 2 / np.sqrt(10)
    for (ba, sa), (bb, sb) in itertools.combinations(pts.items(), 2):
        if np.isclose(abs(sa - sb), d_min):
            assert sum(x != y for x, y in zip(ba, bb)) == 1, (ba, bb)
    for bad in ([0, 1, 1], [[0, 1, 1, 0]], [0, 1, 2, 0]):
        try:
            qam16_mod(bad)
            raise AssertionError("phải báo lỗi")
        except ValueError:
            pass
    print("[OK] qam16_mod (16 điểm, Es=1, Gray)")


def test_sfbc():
    x = sfbc_encode(qam16_mod([0, 1, 1, 0, 1, 0, 0, 1]))
    assert x.shape == (2, 2)
    assert np.allclose(x[:, 0] * np.sqrt(20), [-1 + 3j, 3 - 1j])
    assert np.allclose(x[:, 1] * np.sqrt(20), [-3 - 1j, -1 - 3j])

    rng = np.random.default_rng(0)
    s = qam16_mod(gen_bit(4 * 20_000, rng))
    X = sfbc_encode(s[: 2 * 10_000])
    assert np.isclose(np.mean(np.sum(np.abs(X) ** 2, axis=0)), 1.0, atol=0.02)
    assert np.isclose(np.mean(np.abs(X[0]) ** 2), 0.5, atol=0.02)

    # khớp hệ y = G s trên kênh phẳng, không nhiễu
    H = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    X = sfbc_encode(s[:2])
    Ya, Yb = H @ X[:, 0], H @ X[:, 1]
    y = np.array([Ya[0], Ya[1], np.conj(Yb[0]), np.conj(Yb[1])])
    G = np.array([[H[0, 0], H[0, 1]],
                  [H[1, 0], H[1, 1]],
                  [np.conj(H[0, 1]), -np.conj(H[0, 0])],
                  [np.conj(H[1, 1]), -np.conj(H[1, 0])]]) / np.sqrt(2)
    assert np.allclose(y, G @ s[:2])
    assert np.allclose(G.conj().T @ G, (np.sum(np.abs(H) ** 2) / 2) * np.eye(2))
    print("[OK] sfbc_encode (ví dụ số, công suất, y = G s, GᴴG = c I)")


def run_flow(seed=2026):
    bits = gen_bit(N_BITS, rng=seed)
    s = qam16_mod(bits)
    X = sfbc_encode(s)
    print(f"\nFlow: {bits.size} bit -> {s.size} ký hiệu -> lưới phát {X.shape}")
    print("8 bit đầu :", bits[:8])
    print("s1, s2    :", np.round(s[:2] * np.sqrt(10), 3), "(đơn vị 1/sqrt10)")
    print("Anten 1   :", np.round(X[0, :2] * np.sqrt(20), 3), "(đơn vị 1/sqrt20)")
    print("Anten 2   :", np.round(X[1, :2] * np.sqrt(20), 3))
    print(f"Công suất tổng/bin = {np.mean(np.sum(np.abs(X)**2, axis=0)):.4f}")


if __name__ == "__main__":
    test_gen_bit()
    test_qam16()
    test_sfbc()
    run_flow()
    print("\nTất cả test đạt.")