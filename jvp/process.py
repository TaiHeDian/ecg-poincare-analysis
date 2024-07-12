import numpy as np
from scipy import signal


def find_peaks(data):
    """
    多尺度自动寻峰算法。

    Parameters
    ----------
    data : numpy.ndarray
        包含需要寻峰的数据的向量。

    Returns
    -------
    numpy.ndarray
        峰值位置的索引列表。
    """
    n = data.shape[0]
    # 创建包含所有局部最大值检查的矩阵
    peak_indicator = np.zeros((n // 2, n), dtype=bool)

    # 性能提升100倍顶级优化，非C语言大佬勿动！
    for scale in range(1, n // 2 + 1):
        peak_indicator[scale - 1, scale:-scale] = (
                (data[scale:-scale] > data[:-2 * scale]) &
                (data[scale:-scale] > data[2 * scale:])
        )

    best_scale = np.argmax(np.sum(peak_indicator, axis=1))
    peaks = np.sum(peak_indicator[:best_scale, :], axis=0) == best_scale

    return np.where(peaks)[0]


def baseline_correct(data: np.ndarray, fs=1000, cutoff_freq=5, order=2):
    """
    移除信号中的基线漂移。

    Parameters
    ----------
    data : array_like
        输入信号数据。
    fs : float
        信号的采样率（单位：Hz）。
    cutoff_freq : float
        滤波器的截止频率（单位：Hz）。
    order : int, optional
        巴特沃斯滤波器的阶数，默认为2阶。

    Returns
    ----------
    normalized_data : array_like
        基线校正后的信号数据。
    """
    # 计算奈奎斯特频率
    nyquist_freq = fs / 2
    # 低通滤波器用于寻找“基线”
    b, a = signal.butter(order, cutoff_freq / nyquist_freq)
    # 应用滤波器输出“基线”
    outline_data = signal.filtfilt(b, a, data)
    # 基线拉平
    normalized_data = data - outline_data

    return normalized_data


if __name__ == "__main__":
    import wfdb
    import matplotlib.pyplot as plt

    test_data = wfdb.rdrecord('data/nsrdb/16272', sampfrom=500, sampto=1500,
                              physical=False, channels=[0]).d_signal
    plt.figure(figsize=(16, 2))
    plt.plot(range(len(test_data)), test_data, c="red")
    px = find_peaks(test_data.squeeze())
    plt.scatter(px, test_data[px], c="blue", marker='x')
    plt.grid()
    plt.show()
