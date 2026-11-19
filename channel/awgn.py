import numpy as np


def ebn0_to_n0(ebn0_db, bits_per_symbol=4, es=1.0):
    """
    Eb/N0 hữu ích (dB) -> N0  (kế hoạch mục 2.2).
    Eb = Es / bits_per_symbol, với 16-QAM và Es = 1:  N0 = 1 / (4 * gamma_b).
    Không tính năng lượng CP (nếu tính thì cộng 0.97 dB vào nhãn Eb/N0).
    """
    gamma_b = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    return es / (bits_per_symbol * gamma_b)


def awgn(shape, n0, rng=None):
    """Nhiễu Gauss phức sqrt(N0/2) * (u + jv), u, v ~ N(0,1)  ->  E|n|^2 = N0."""
    if n0 < 0:
        raise ValueError("n0 phải >= 0")
    rng = np.random.default_rng(rng)
    return np.sqrt(n0 / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
