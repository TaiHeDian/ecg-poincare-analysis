import matplotlib.pyplot as plt

from jvp_read import read_data


def plot_dataframes(data_dict):
    # 创建一个大的图形窗口，包含多个子图
    # 调整子图的行数和列数以及整体图形大小
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
    
    # 将axes数组展平，便于迭代
    axes = axes.flatten()
    
    # 遍历字典中的数据框，绘制每个数据框的图形
    for ax, (filename, df) in zip(axes, data_dict.items()):
        # 绘制折线图
        ax.plot(df['Time/sec'], df['Current/A'])
        ax.set_title(filename)  # 使用文件名作为子图标题
        ax.set_xlabel('Time (sec)')  # 设置X轴标签
        ax.set_ylabel('Current (A)')  # 设置Y轴标签
    
    # 自动调整子图间距
    plt.tight_layout()
    plt.show()

# 假设experiment_data是你从函数中读取的字典，包含了多个DataFrame
plot_dataframes(read_data('datas/jvp/commercial/'))
