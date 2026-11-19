import numpy as np

from channel_profiles import get_profile


def freq_response(h, delays, n_fft=64):
    """
    H[..., k] = sum_l h[..., l] * exp(-j 2 pi k d_l / N)
    Tương đương FFT THÔNG THƯỜNG (không chuẩn hóa) của đáp ứng xung đệm 0 tới N mẫu.
    """
    h = np.asarray(h, dtype=complex)
    delays = np.asarray(delays, dtype=int)
    if h.shape[-1] != delays.size:
        raise ValueError("trục cuối của h phải bằng số tap")
    if delays.min() < 0 or delays.max() >= n_fft:
        raise ValueError("độ trễ tap phải nằm trong [0, n_fft)")
    k = np.arange(n_fft)
    return h @ np.exp(-2j * np.pi * np.outer(delays, k) / n_fft)


def rayleigh_mimo_channel(profile="main", n_blocks=None, nr=2, nt=2, n_fft=64, rng=None):
    """
    Sinh kênh Rayleigh đa đường cho nr x nt liên kết (CSI lý tưởng tại máy thu).
    h_ij[l] ~ CN(0, P_l), độc lập giữa tap, liên kết và khối OFDM.
    Kênh cố định trong một khối, kể cả CP (block fading).

    Trả về:
      h      : (nr, nt, L)      hoặc (n_blocks, nr, nt, L)
      H      : (nr, nt, n_fft)  hoặc (n_blocks, nr, nt, n_fft)   -> H[rx, tx, k]
      delays : (L,) độ trễ các tap (mẫu)
    """
    if n_blocks is not None and (not isinstance(n_blocks, (int, np.integer)) or n_blocks <= 0):
        raise ValueError("n_blocks phải là số nguyên dương hoặc None")
    delays, powers = get_profile(profile)
    rng = np.random.default_rng(rng)
    lead = () if n_blocks is None else (n_blocks,)
    shape = lead + (nr, nt, delays.size)
    g = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
    h = g * np.sqrt(powers)
    return h, freq_response(h, delays, n_fft), delays
