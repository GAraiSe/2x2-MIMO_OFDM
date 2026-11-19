import numpy as np


# Profile công suất trễ (kế hoạch mục 2.2). Mỗi liên kết có tổng công suất trung bình = 1.
PROFILES = {
    "main":       {"delays": [0, 1, 2, 3],   "powers": [0.25, 0.25, 0.25, 0.25]},  # kênh chính
    "flat":       {"delays": [0],            "powers": [1.0]},                     # Rayleigh phẳng
    "long_delay": {"delays": [0, 5, 10, 15], "powers": [0.25, 0.25, 0.25, 0.25]},  # chọn lọc mạnh
}


def get_profile(name="main"):
    """Trả về (delays, powers): độ trễ tap (mẫu) và công suất tap đã chuẩn hóa tổng = 1."""
    if name not in PROFILES:
        raise ValueError(f"profile phải là một trong {list(PROFILES)}")
    delays = np.asarray(PROFILES[name]["delays"], dtype=int)
    powers = np.asarray(PROFILES[name]["powers"], dtype=float)
    return delays, powers / powers.sum()


def adjacent_corr(name="main", n_fft=64, dk=1):
    """
    Độ lớn tương quan lý thuyết giữa hai sóng mang cách nhau dk:
        |R(dk)| = |sum_l P_l exp(-j 2 pi dk d_l / N)|
    dk = 1 là cặp sóng mang của SFBC: 'main' -> 0.994, 'long_delay' -> 0.855.
    """
    delays, powers = get_profile(name)
    return float(np.abs(np.sum(powers * np.exp(-2j * np.pi * dk * delays / n_fft))))
