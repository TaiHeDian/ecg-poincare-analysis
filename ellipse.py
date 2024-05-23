import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as stats

def plot_confidence_ellipse(data, conf_level=0.95):
    """
    Plot a confidence ellipse over a 2D scatter data.

    Parameters:
    - data (np.array): Input 2D data points as a Numpy array of shape (n, 2).
    - conf_level (float): Confidence level for the ellipse. Default is 0.95.

    Returns:
    - tuple: (center_x, center_y, major_axis, minor_axis)
             Center coordinates and lengths of major and minor axes.
    """
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

    # 绘制数据点
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=5)

    # 绘制椭圆
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                      edgecolor='red', fc='None', lw=2, linestyle='--')
    ax.add_patch(ellipse)

    # 设置图形
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()

    return mean[0], mean[1], width, height

# Example usage
if __name__ == "__main__":
    # Generate some example data
    data = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], size=300)
    center_x, center_y, major_axis, minor_axis = plot_confidence_ellipse(data)
    print("Ellipse Center: ({}, {})".format(center_x, center_y))
    print("Major Axis Length:", major_axis)
    print("Minor Axis Length:", minor_axis)
