import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from readdata import read_data


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
        ax.set_xlim(left=0, right=value[:, 0].max())

    # 隐藏多余的子图
    for ax in axs[n:]:
        ax.axis('off')

    # 自动调整子图间距
    plt.tight_layout()

    # 保存图片
    # 获取当前日期和时间
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')  # 格式化日期时间字符串

    # 创建文件名和保存路径
    folder_path = '../out/'
    filename = f"plot_{formatted_time}.png"
    save_path = os.path.join(folder_path, filename)

    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    plot_dict_data(read_data('../data/jvp/commercial/'))
    plt.show()
