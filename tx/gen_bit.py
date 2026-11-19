import numpy as np


def gen_bit(n_bits, rng=None):
    if (
        isinstance(n_bits, (bool, np.bool_))
        or not isinstance(n_bits, (int, np.integer))
        or n_bits <= 0
        or n_bits % 4
    ):
        raise ValueError("n_bits phải là số nguyên dương chia hết cho 4")
    rng = np.random.default_rng(rng)
    bits = rng.integers(0, 2, size=n_bits, dtype=np.int8)
    return bits

