import numpy as np

def sfbc_encode(s):
    s = np.asarray(s, dtype=complex)
    if s.ndim != 1:
        raise ValueError("s phải là mảng 1-D")
    if len(s) % 2 != 0:
        raise ValueError("Số ký hiệu phải chẵn (ghép theo cặp sóng mang)")
    s1 = s[0::2]         
    s2 = s[1::2]          
    X = np.empty((2, len(s)), dtype=complex)   
    X[0, 0::2] =  s1
    X[0, 1::2] = -np.conj(s2)
    X[1, 0::2] =  s2
    X[1, 1::2] =  np.conj(s1)

    return X / np.sqrt(2)