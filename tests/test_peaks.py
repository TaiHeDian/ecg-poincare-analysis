import matplotlib.pyplot as plt
import wfdb

from peaks import find_peaks

test_data = wfdb.rdrecord("../data/nsrdb/16272", sampfrom=500, sampto=1500,
                          physical=False, channels=[0]).d_signal
plt.figure(figsize=(16, 2))
plt.plot(range(len(test_data)), test_data, c="red")
px = find_peaks(test_data.squeeze())
plt.scatter(px, test_data[px], c="blue", marker='x')
plt.grid()
plt.show()
