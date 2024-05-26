import numpy as np

def ampd(data):
    """
    Automatic multiscale-based peak detection Algorithm.

    Parameters
    ----------
    data : numpy.ndarray
        1-dimensional vector containing input data. 

    Returns
    -------
    numpy.ndarray
        List of index values where the peaks are located.
    """
    n = data.shape[0]
    # Create a 2D array to contain all local maxima checks
    peak_indicator = np.zeros((n // 2, n), dtype=bool)

    for scale in range(1, n // 2 + 1):
        peak_indicator[scale - 1, scale:-scale] = (
            (data[scale:-scale] > data[:-2 * scale]) &
            (data[scale:-scale] > data[2 * scale:])
        )

    best_scale = np.argmax(np.sum(peak_indicator, axis=1))
    P = np.sum(peak_indicator[:best_scale, :], axis=0) == best_scale

    return np.where(P)[0]

if __name__ == "__main__":
    import wfdb
    import matplotlib.pyplot as plt
    
    data = wfdb.rdrecord('datas/nsrdb/16272', sampfrom=500, sampto=1500, physical=False, channels=[0]).d_signal
    plt.figure(figsize=(16, 2))
    plt.plot(range(len(data)), data, c="red")
    px = ampd(data.squeeze())
    plt.scatter(px, data[px], c="blue", marker='x')
    plt.grid()
    plt.show()
