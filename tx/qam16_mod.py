import numpy as np


LUT = np.array([-3, -1, 3, 1])


def qam16_mod(bits):
    """
    Gray 16-QAM modulation:
    Mapping per axis: 00, 01, 10, 11 -> -3, -1, +3, +1.
    Four bits map to one complex symbol.
    """
    bits = np.asarray(bits)
    if bits.ndim != 1 or bits.size % 4 or not np.isin(bits, (0, 1)).all():
        raise ValueError("bits: Chuỗi bit chỉ 0-1 và độ dài chia hết cho 4")

    g = bits.astype(int).reshape(-1, 4)
    I = LUT[2*g[:, 0] + g[:, 1]]
    Q = LUT[2*g[:, 2] + g[:, 3]]
    return (I + 1j * Q) / np.sqrt(10)

