import matplotlib.pyplot as plt
import numpy as np
import wfdb

from jvp.peaks import find_peaks
from jvp.poincare import poincare

signal_length = 20000

# 获取并处理不规则异常心跳数据
record_arrhythmia = wfdb.rdrecord('../data/mitdb/101', sampto=signal_length, channels=[0])
arrhythmia_data = record_arrhythmia.p_signal
arrhythmia_peaks = find_peaks(arrhythmia_data.squeeze())
arrhythmia_intervals = np.column_stack((np.diff(arrhythmia_peaks[:-1]), np.diff(arrhythmia_peaks[1:])))

# 获取并处理正常心跳数据
record_normal = wfdb.rdrecord('../data/nsrdb/16272', sampto=signal_length, channels=[0])
normal_data = record_normal.p_signal
normal_peaks = find_peaks(normal_data.squeeze())
normal_intervals = np.column_stack((np.diff(normal_peaks[:-1]), np.diff(normal_peaks[1:])))

# Plotting
plt.figure(figsize=(14, 7))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

# Plot for Arrhythmia Data
ax1.scatter(arrhythmia_intervals[:, 0], arrhythmia_intervals[:, 1], edgecolors='blue', c='red')
poincare(ax1, arrhythmia_intervals, conf_level=0.9, edge_color='blue')
ax1.set_title('Poincare Plot for Arrhythmia')
ax1.set_xlabel('NN Interval [n]')
ax1.set_ylabel('NN Interval [n+1]')
ax1.set_xlim(np.min(arrhythmia_intervals[:, 0]) - 50, np.percentile(arrhythmia_intervals[:, 0], 95) + 50)
ax1.set_ylim(np.min(arrhythmia_intervals[:, 1]) - 50, np.percentile(arrhythmia_intervals[:, 1], 95) + 50)

std_ratio_arrhythmia = np.std(np.diff(arrhythmia_intervals)) / np.sqrt(2)
plt.figtext(0.07, 0.9, f"STD = {std_ratio_arrhythmia:.2f}", fontsize=12, color='blue')

# Plot for Normal Heartbeat Data
ax2.scatter(normal_intervals[:, 0], normal_intervals[:, 1], edgecolors='red', c='blue')
poincare(ax2, normal_intervals, conf_level=0.9, edge_color='red')
ax2.set_title('Poincare Plot for Normal Heartbeat')
ax2.set_xlabel('NN Interval [n]')
ax2.set_ylabel('NN Interval [n+1]')
ax2.set_xlim(np.min(normal_intervals[:, 0] - 10), np.percentile(normal_intervals[:, 0], 95) + 10)
ax2.set_ylim(np.min(normal_intervals[:, 1] - 10), np.percentile(normal_intervals[:, 1], 95) + 10)

std_ratio_normal = np.std(np.diff(normal_intervals)) / np.sqrt(2)
plt.figtext(0.57, 0.1, f"STD = {std_ratio_normal:.2f}", fontsize=12, color='red')

plt.tight_layout()
plt.show()
