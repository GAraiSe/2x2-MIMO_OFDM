import numpy as np

from awgn import awgn


def channel_freq(X, H, n0=0.0, rng=None):
    """
    Mô hình tương đương sau FFT (đúng khi N_CP >= độ trễ lớn nhất):
        Y[rx, k] = sum_tx H[rx, tx, k] * X[tx, k] + N[rx, k]
    X : (NT, N) hoặc (nb, NT, N)  - lưới phát từ sfbc_encode
    H : (NR, NT, N) hoặc (nb, NR, NT, N)
    ->  Y : (NR, N) hoặc (nb, NR, N)
    """
    X = np.asarray(X, dtype=complex)
    H = np.asarray(H, dtype=complex)
    if H.shape[-2] != X.shape[-2] or H.shape[-1] != X.shape[-1]:
        raise ValueError("H phải có dạng (..., NR, NT, N) khớp với X (..., NT, N)")
    Y = np.einsum('...rtk,...tk->...rk', H, X)
    return Y + awgn(Y.shape, n0, rng) if n0 > 0 else Y


def channel_time(x, h, delays, n0=0.0, rng=None):
    """
    Tích chập tuyến tính trong miền thời gian rồi cộng AWGN độc lập ở mỗi anten thu:
        y[rx, n] = sum_tx sum_l h[rx, tx, l] * x[tx, n - d_l] + w[rx, n]
    x : (..., NT, Ns) chuỗi mẫu phát đã chèn CP (có thể nối nhiều khối OFDM)
    h : (..., NR, NT, L) kênh giữ cố định trên cả chuỗi x
    ->  y : (..., NR, Ns)  (bỏ phần đuôi tích chập vượt quá Ns mẫu)
    """
    x = np.asarray(x, dtype=complex)
    h = np.asarray(h, dtype=complex)
    delays = np.asarray(delays, dtype=int)
    if h.shape[-2] != x.shape[-2] or h.shape[-1] != delays.size:
        raise ValueError("h phải có dạng (..., NR, NT, L) khớp với x (..., NT, Ns)")
    ns = x.shape[-1]
    y = np.zeros(x.shape[:-2] + (h.shape[-3], ns), dtype=complex)
    for l, d in enumerate(delays):
        xs = np.zeros_like(x)
        xs[..., d:] = x[..., :ns - d]          # trễ d mẫu
        y += np.einsum('...rt,...tn->...rn', h[..., l], xs)
    return y + awgn(y.shape, n0, rng) if n0 > 0 else y
