from scipy import signal

def baseline_correct(data, fs=1000, cutoff_freq=5, order=2):
    """
    移除信号中的基线漂移。

    参数:
    data : array_like
        输入信号数据。
    fs : float
        信号的采样率（单位：Hz）。
    cutoff_freq : float
        滤波器的截止频率（单位：Hz）。
    order : int, optional
        巴特沃斯滤波器的阶数，默认为2阶。

    返回:
    normalized_data : ndarray
        基线校正后的信号数据。
    """
    # 计算奈奎斯特频率
    nyquist_freq = fs / 2
    # 设计低通滤波器
    b, a = signal.butter(order, cutoff_freq / nyquist_freq)
    # 应用滤波器
    outline_data = signal.filtfilt(b, a, data)
    # 进行基线拉平
    normalized_data = data - outline_data

    return normalized_data
