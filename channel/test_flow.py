import os
import sys
import numpy as np

from channel_profiles import PROFILES, get_profile, adjacent_corr
from rayleigh_mimo_channel import rayleigh_mimo_channel
from awgn import ebn0_to_n0, awgn
from apply_channel import channel_freq, channel_time

N_FFT = 64
N_CP = 16


# IFFT/FFT đơn vị + CP tối giản, CHỈ để kiểm thử kênh.
# Khi có ofdm_tx (TV2) và ofdm_rx (TV4) thì thay bằng hàm chính thức.
def _ofdm_tx(X, ncp):
    x = np.fft.ifft(X, axis=-1, norm="ortho")
    return np.concatenate([x[..., -ncp:], x], axis=-1) if ncp > 0 else x


def _ofdm_rx(y, ncp, n_fft):
    return np.fft.fft(y[..., ncp:ncp + n_fft], axis=-1, norm="ortho")


def test_profiles():
    for name in PROFILES:
        d, p = get_profile(name)
        assert np.isclose(p.sum(), 1.0) and d.max() <= N_CP
    assert np.isclose(adjacent_corr("main"), 0.994, atol=1e-3)
    assert np.isclose(adjacent_corr("long_delay"), 0.855, atol=1e-3)
    assert np.isclose(adjacent_corr("flat"), 1.0)
    try:
        get_profile("abc")
        raise AssertionError("phải báo lỗi")
    except ValueError:
        pass
    print("[OK] profile kênh (tổng công suất = 1, tương quan kề 0.994 / 0.855)")


def test_channel_stats():
    nb = 20_000
    h1, H1, _ = rayleigh_mimo_channel("main", rng=7)
    h2, _, _ = rayleigh_mimo_channel("main", rng=7)
    assert h1.shape == (2, 2, 4) and H1.shape == (2, 2, N_FFT)
    assert np.array_equal(h1, h2), "cùng seed phải cho cùng kênh"

    for name in PROFILES:
        h, H, _ = rayleigh_mimo_channel(name, n_blocks=nb, rng=1)
        _, p = get_profile(name)
        assert np.allclose(np.mean(np.abs(h) ** 2, axis=(0, 1, 2)), p, rtol=0.03), name
        assert abs(np.mean(np.abs(H) ** 2) - 1) < 0.02, name

    _, H, _ = rayleigh_mimo_channel("main", n_blocks=nb, rng=2)
    a = np.abs(H).ravel()
    assert abs(a.mean() - np.sqrt(np.pi) / 2) < 0.01, "E|H| phải ~ sqrt(pi)/2 (Rayleigh)"
    assert abs(np.mean(a ** 2 < 0.1) - (1 - np.exp(-0.1))) < 0.005, "P(fading sâu) sai"
    C = np.abs(np.corrcoef(H[:, :, :, 0].reshape(nb, 4).T))
    assert np.max(C - np.eye(4)) < 0.03, "4 liên kết phải độc lập"
    print(f"[OK] thống kê kênh (E|H| = {a.mean():.4f} ~ 0.8862, 4 liên kết độc lập)")


def test_freq_corr():
    for name in PROFILES:
        _, H, _ = rayleigh_mimo_channel(name, n_blocks=20_000, rng=3)
        rho = np.abs(np.mean(H[..., :-1] * np.conj(H[..., 1:]))) / np.mean(np.abs(H) ** 2)
        assert abs(rho - adjacent_corr(name)) < 0.01, name
    print("[OK] tương quan hai sóng mang kề khớp lý thuyết")


def test_awgn():
    n0 = ebn0_to_n0(10.0)
    assert np.isclose(n0, 1 / 40)
    assert np.allclose(ebn0_to_n0([0, 20]), [0.25, 0.0025])
    w = awgn((2000, N_FFT), n0, rng=4)
    W = np.fft.fft(w, axis=-1, norm="ortho")
    assert abs(np.mean(np.abs(w) ** 2) / n0 - 1) < 0.01, "E|n|^2 phải = N0"
    assert abs(np.mean(np.abs(W) ** 2) / n0 - 1) < 0.01, "sau FFT ortho vẫn = N0"
    assert abs(np.var(w.real) / np.var(w.imag) - 1) < 0.02, "I/Q mỗi trục N0/2"
    assert np.all(awgn((3,), 0.0) == 0)
    print("[OK] AWGN (N0 = 1/(4 gamma_b), phương sai N0 ở cả hai miền)")


def _compare_domains(profile, ncp, n_blocks=8, seed=5):
    """Nối n_blocks khối OFDM qua CÙNG một kênh (miền thời gian) rồi so với H[k] X[k]."""
    rng = np.random.default_rng(seed)
    h, H, d = rayleigh_mimo_channel(profile, rng=rng)
    X = (rng.standard_normal((n_blocks, 2, N_FFT))
         + 1j * rng.standard_normal((n_blocks, 2, N_FFT))) / np.sqrt(2)
    x = np.moveaxis(_ofdm_tx(X, ncp), 0, -2).reshape(2, -1)      # (NT, n_blocks*(N+CP))
    y = channel_time(x, h, d)
    Y_t = _ofdm_rx(np.moveaxis(y.reshape(2, n_blocks, -1), 1, 0), ncp, N_FFT)
    Y_f = channel_freq(X, H[None])
    err = np.abs(Y_t - Y_f)[1:]                 # bỏ khối đầu (không có khối trước gây ISI)
    nmse = np.sum(err ** 2) / np.sum(np.abs(Y_f[1:]) ** 2)
    return err.max(), 10 * np.log10(nmse + 1e-300)


def test_time_vs_freq():
    for name in PROFILES:
        e, _ = _compare_domains(name, N_CP)
        assert e < 1e-12, f"{name}: CP đủ dài thì hai miền phải trùng ({e:.1e})"
    e1, n1 = _compare_domains("main", 2)
    e2, n2 = _compare_domains("long_delay", 8)
    assert e1 > 1e-3 and e2 > 1e-3, "CP ngắn hơn trễ lớn nhất thì phải lệch"
    print(f"[OK] miền thời gian = miền tần số khi CP = {N_CP} (sai lệch < 1e-12); "
          f"CP ngắn -> NMSE {n1:.1f} dB / {n2:.1f} dB")


def test_with_tx():
    """Nối với code TV2 (tx/): bit -> 16QAM -> SFBC -> IFFT+CP -> kênh -> bỏ CP+FFT -> y = G s."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tx"))
    from gen_bit import gen_bit
    from qam16_mod import qam16_mod
    from sfbc_encode import sfbc_encode

    s = qam16_mod(gen_bit(N_FFT * 4, rng=2026))
    X = sfbc_encode(s)                                   # (2, 64), đã gồm 1/sqrt(2)
    h, H, d = rayleigh_mimo_channel("main", rng=2026)
    Y = _ofdm_rx(channel_time(_ofdm_tx(X, N_CP), h, d), N_CP, N_FFT)
    assert np.allclose(Y, channel_freq(X, H)), "khác mô hình miền tần số"

    # Hệ tương đương cho từng cặp (a, b) = (2p, 2p+1), kế hoạch mục 2.3 — đầu vào cho TV4
    for p in range(N_FFT // 2):
        a, b = 2 * p, 2 * p + 1
        Ha, Hb = H[:, :, a], H[:, :, b]
        y = np.array([Y[0, a], Y[1, a], np.conj(Y[0, b]), np.conj(Y[1, b])])
        G = np.array([[Ha[0, 0], Ha[0, 1]],
                      [Ha[1, 0], Ha[1, 1]],
                      [np.conj(Hb[0, 1]), -np.conj(Hb[0, 0])],
                      [np.conj(Hb[1, 1]), -np.conj(Hb[1, 0])]]) / np.sqrt(2)
        assert np.allclose(y, G @ s[a:b + 1]), f"cặp {p}: y != G s"
    print("[OK] nối với tx/: bit -> SFBC -> kênh 2x2 miền thời gian -> y = G s đúng trên 32 cặp")


def run_flow():
    print("\nQuy đổi Eb/N0 -> N0 (Es = 1, 16-QAM):")
    print("  Eb/N0 (dB) | Es/N0 (dB) |     N0")
    for e in np.arange(0, 25.1, 2.5):
        print(f"     {e:5.1f}   |   {e + 6.02:5.2f}    | {ebn0_to_n0(e):.4e}")
    h, H, d = rayleigh_mimo_channel("main", rng=2026)
    print(f"\nKênh 'main': h {h.shape}, H {H.shape}, trễ {d.tolist()}")
    print("|H11[k]| 8 bin đầu:", np.round(np.abs(H[0, 0, :8]), 3))


if __name__ == "__main__":
    test_profiles()
    test_channel_stats()
    test_freq_corr()
    test_awgn()
    test_time_vs_freq()
    test_with_tx()
    run_flow()
    print("\nTất cả test đạt.")
