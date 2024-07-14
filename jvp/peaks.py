import numpy as np


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

    for scale in range(1, n // 2 + 1):
        peak_indicator[scale - 1, scale:-scale] = (
                (data[scale:-scale] > data[:-2 * scale]) &
                (data[scale:-scale] > data[2 * scale:])
        )

    best_scale = np.argmax(np.sum(peak_indicator, axis=1))
    peaks = np.sum(peak_indicator[:best_scale, :], axis=0) == best_scale

    return np.where(peaks)[0]
