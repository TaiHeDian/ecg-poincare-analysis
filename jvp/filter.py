import numpy as np
from scipy import signal


def baseline_correct(data: np.ndarray, fs=1000, cutoff=5, order=2):
    """
    移除信号中的基线漂移。

    Parameters
    ----------
    data : array_like
        输入信号数据。
    fs : float
        信号的采样率（单位：Hz）。
    cutoff : float
        滤波器的截止频率（单位：Hz）。
    order : int, optional
        巴特沃斯滤波器的阶数，默认为2阶。

    Returns
    ----------
    filtered_data : array_like
        基线校正后的信号数据。
    """
    # 计算奈奎斯特频率
    nyquist_freq = fs / 2
    # 应用低通滤波器寻找并输出“基线”
    b, a = signal.butter(order, cutoff / nyquist_freq)
    outline_data = signal.filtfilt(b, a, data)
    # 基线拉平
    filtered_data = data - outline_data

    return filtered_data
