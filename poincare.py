import numpy as np
import wfdb
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as stats

from ampd import ampd

# Helper function to calculate differences
def calculate_differences(peaks):
    intervals = np.diff(peaks)
    return intervals, np.diff(intervals), intervals[:-1] + intervals[1:]

def plot_confidence_ellipse(ax, data, conf_level=0.95, edge_color='red'):
    # 计算数据的均值和协方差矩阵
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)

    # 特征分解
    vals, vecs = np.linalg.eigh(cov)
    
    # 计算椭圆的轴长
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # 计算椭圆的角度
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # 椭圆的宽度和高度
    width, height = 2 * np.sqrt(vals * stats.chi2.ppf(conf_level, 2))

    # 绘制椭圆
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                      edgecolor=edge_color, fc='None', lw=2, linestyle='--')
    ax.add_patch(ellipse)

signal_length = 75000

# 获取并处理不规则异常心跳数据
record_arrhythmia = wfdb.rdrecord('101', pn_dir='mitdb', sampto=signal_length, physical=False, channels=[0])
arrhythmia_data = record_arrhythmia.d_signal
arrhythmia_peaks = ampd(arrhythmia_data.squeeze())
arrhythmia_intervals, delta_arrhythmia, sum_arrhythmia = calculate_differences(arrhythmia_peaks)

# 获取并处理正常心跳数据
record_normal = wfdb.rdrecord('16272', pn_dir='nsrdb', sampto=signal_length, physical=False, channels=[0])
normal_data = record_normal.d_signal
normal_peaks = ampd(normal_data.squeeze())
normal_intervals, delta_normal, sum_normal = calculate_differences(normal_peaks)

# Plotting
plt.figure(figsize=(14, 7))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

# Plot for Arrhythmia Data
ax1.scatter(arrhythmia_intervals[:-1], arrhythmia_intervals[1:], edgecolors='blue', c='red', label='Arrhythmia')
ax1.set_xlabel('NN Interval [n]')
ax1.set_ylabel('NN Interval [n+1]')
ax1.set_title('Poincare Plot for Arrhythmia')

# Plot for Normal Heartbeat Data
ax2.scatter(normal_intervals[:-1], normal_intervals[1:], edgecolors='red', c='blue', label='Normal Heartbeat')
ax2.set_xlabel('NN Interval [n]')
ax2.set_ylabel('NN Interval [n+1]')
ax2.set_title('Poincare Plot for Normal Heartbeat')

plt.tight_layout()
plt.show()

# Calculate and print standard deviation ratios
std_ratio_arrhythmia = np.std(delta_arrhythmia) / np.sqrt(2)
std_ratio_normal = np.std(delta_normal) / np.sqrt(2)

print('STD for Arrhythmia:', std_ratio_arrhythmia)
print('STD for Normal Heartbeat:', std_ratio_normal)
