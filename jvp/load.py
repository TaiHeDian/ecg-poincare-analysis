import os

import numpy as np


def read_data(directory):
    """读取 CHI660e 输出的 I-t 文本数据。

    Parameters
    ----------
    directory : `str`
        包含要读取文件的路径。

    Returns
    -------
    `dict`
        包含文件名及其对应数据数组的字典。

        2维 Numpy 数组，两列值对应时间和电流。
    """
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path) as file:
                for nu, line in enumerate(file):
                    if 'Time/sec, Current/A' in line:
                        header_line = nu
                        break
                data = np.loadtxt(file_path, delimiter=',', skiprows=header_line + 1)
                results[filename] = data
    return results


if __name__ == '__main__':
    data_dict = read_data('../data/jvp/commercial/')
    print(data_dict.keys())
