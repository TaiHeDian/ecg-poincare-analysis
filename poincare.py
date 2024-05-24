import numpy as np
import wfdb
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as stats

from ampd import ampd


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

    ax.scatter(mean[0], mean[1], c=edge_color, marker='x',
               label=f'({mean[0]:.2f}, {mean[1]:.2f})\n\nSD1={width:.2f}\nSD2={height:.2f}'
               )
    ax.legend()

    return mean[0], mean[1], width, height


signal_length = 20000

# 获取并处理不规则异常心跳数据
record_arrhythmia = wfdb.rdrecord('datas/mitdb/101', sampto=signal_length, channels=[0])
arrhythmia_data = record_arrhythmia.p_signal
arrhythmia_peaks = ampd(arrhythmia_data.squeeze())
arrhythmia_intervals = np.column_stack((np.diff(arrhythmia_peaks[:-1]), np.diff(arrhythmia_peaks[1:])))

# 获取并处理正常心跳数据
record_normal = wfdb.rdrecord('datas/nsrdb/16272', sampto=signal_length, channels=[0])
normal_data = record_normal.p_signal
normal_peaks = ampd(normal_data.squeeze())
normal_intervals = np.column_stack((np.diff(normal_peaks[:-1]), np.diff(normal_peaks[1:])))

# Plotting
plt.figure(figsize=(14, 7))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

# Plot for Arrhythmia Data
ax1.scatter(arrhythmia_intervals[:, 0], arrhythmia_intervals[:, 1], edgecolors='blue', c='red')
plot_confidence_ellipse(ax1, arrhythmia_intervals, conf_level=0.9, edge_color='blue')
ax1.set_title('Poincare Plot for Arrhythmia')
ax1.set_xlabel('NN Interval [n]')
ax1.set_ylabel('NN Interval [n+1]')
ax1.set_xlim(np.min(arrhythmia_intervals[:, 0]) - 50, np.percentile(arrhythmia_intervals[:, 0], 95) + 50)
ax1.set_ylim(np.min(arrhythmia_intervals[:, 1]) - 50, np.percentile(arrhythmia_intervals[:, 1], 95) + 50)

std_ratio_arrhythmia = np.std(np.diff(arrhythmia_intervals)) / np.sqrt(2)
plt.figtext(0.07, 0.9, f"STD = {std_ratio_arrhythmia:.2f}", fontsize=12, color='blue')

# Plot for Normal Heartbeat Data
ax2.scatter(normal_intervals[:, 0], normal_intervals[:, 1], edgecolors='red', c='blue')
plot_confidence_ellipse(ax2, normal_intervals, conf_level=0.9, edge_color='red')
ax2.set_title('Poincare Plot for Normal Heartbeat')
ax2.set_xlabel('NN Interval [n]')
ax2.set_ylabel('NN Interval [n+1]')
ax2.set_xlim(np.min(normal_intervals[:, 0] - 10), np.percentile(normal_intervals[:, 0], 95) + 10)
ax2.set_ylim(np.min(normal_intervals[:, 1] - 10), np.percentile(normal_intervals[:, 1], 95) + 10)

std_ratio_normal = np.std(np.diff(normal_intervals)) / np.sqrt(2)
plt.figtext(0.57, 0.1, f"STD = {std_ratio_normal:.2f}", fontsize=12, color='red')

plt.tight_layout()
plt.show()
