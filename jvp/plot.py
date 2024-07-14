import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from load import read_data
from peaks import find_peaks
from poincare import poincare
from process import filter_dict_data


def generate_filename():
    # 获取当前日期和时间
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')  # 格式化日期时间字符串

    # 创建文件名和保存路径
    folder_path = '../out/'
    filename = f"Plot_{formatted_time}.png"

    return os.path.join(folder_path, filename)


def plot_dict_data(data_dict, cols=3):
    """
    使用包含多个数组的字典绘制网格状图像。

    Parameters
    ----------
    data_dict : `dict`
        包含文件名及其对应数据数组的字典。
    cols : `int`
        绘制图形的列数。默认为3。

    Returns
    -------
    None
    """
    # 计算需要多少行多少列来显示所有图形
    n = len(data_dict)
    rows = (n + cols - 1) // cols  # 计算行数

    # 创建一个大的图形窗口，包含多个子图
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 2))

    # 如果只有一个图，axs不会自动成为数组，需要这样处理
    if n == 1:
        axs = np.array([axs])

    # 压扁axs数组，便于迭代，不管它的原始形状如何
    axs = axs.flatten()

    # 遍历字典中的键值对，绘制每个数据集
    for ax, (key, value) in zip(axs, data_dict.items()):
        # 绘制折线图
        ax.plot(value[:, 0], value[:, 1])
        ax.set_title(key)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Current (A)')
        ax.set_xlim(left=value[:, 0].min(), right=value[:, 0].max())

    # 隐藏多余的子图
    for ax in axs[n:]:
        ax.axis('off')

    # 自动调整子图间距
    plt.tight_layout()

    plt.savefig(generate_filename())
    plt.close()


def plot_single_data(data):
    plt.figure(figsize=(20, 2))
    plt.plot(data[:, 0], data[:, 1], c="red")
    plt.xlabel('Time (sec)')  # 设置X轴标签
    plt.ylabel('Current (A)')  # 设置Y轴标签
    plt.xlim(left=data[:, 0].min(), right=data[:, 0].max())
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)

    plt.tight_layout()

    plt.savefig(generate_filename())
    plt.close()


def plot_peaks(data):
    plt.figure(figsize=(16, 5))
    plt.plot(data[:, 0], data[:, 1], c="red")
    px = find_peaks(data[:, 1].squeeze())
    plt.scatter(data[:, 0][px], data[:, 1][px], c="blue", marker='x')
    plt.xlabel('Time (sec)')  # 设置X轴标签
    plt.ylabel('Current (A)')  # 设置Y轴标签
    plt.xlim(left=data[:, 0].min(), right=data[:, 0].max())
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)

    plt.tight_layout()

    plt.savefig(generate_filename())
    plt.close()


def plot_dict_poincare(data_dict, cols=3):
    n = len(data_dict)  # Number of datasets
    rows = (n + cols - 1) // cols  # Calculate required number of rows

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # Create subplot grid

    if n == 1:
        axs = np.array([axs])

    axs = axs.flatten()

    for ax, (key, data) in zip(axs, data_dict.items()):
        peaks = find_peaks(data[:, 1])
        intervals = np.column_stack((np.diff(peaks[:-1]), np.diff(peaks[1:])))
        ax.set_title(key)
        ax.scatter(intervals[:, 0], intervals[:, 1], edgecolors='red', c='blue')
        poincare(ax, intervals, conf_level=0.9, edge_color='red')  # Call the existing poincare function
        ax.set_xlabel('RRi')
        ax.set_ylabel('RRi+1')
        std_ratio = np.std(np.diff(intervals)) / np.sqrt(2)
        ax.text(0.9, 0.9, f"STD = {std_ratio:.2f}",
                color='blue', ha='right', va='top', transform=ax.transAxes)

    # Hide any unused axes
    for ax in axs[len(data_dict):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(generate_filename())
    plt.close()


if __name__ == '__main__':
    plot_dict_poincare(filter_dict_data(read_data('../data/jvp/commercial/')))
