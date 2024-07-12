import numpy as np
import scipy.stats as stats
from matplotlib.patches import Ellipse


def poincare_plot(ax, data, conf_level=0.95, edge_color='red'):
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
