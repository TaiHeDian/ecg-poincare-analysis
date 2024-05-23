import numpy as np

def ampd(data):
    """
    AMPD算法
    :param data: 1-D numpy.ndarray
    :return: 波峰所在索引值的列表
    """
    count = data.shape[0]
    lmax = count // 2
    # 创建一个二维数组，其中包含所有局部最大的检查
    L = np.zeros((lmax, count), dtype=bool)

    for k in range(1, lmax + 1):
        L[k - 1, k:-k] = (data[k:-k] > data[:-2 * k]) & (data[k:-k] > data[2 * k:])

    # 计算每一行中False的数量（不是局部最大值）
    G = np.sum(~L, axis=1)
    
    # 找到G中最小值的索引
    l = np.argmin(G)

    # 生成从1到l的所有整数的数组
    K = np.arange(1, l + 1)
    # 为每个i计算符合条件的K的数量
    P = np.sum(L[:l, :], axis=0) == l

    # 返回P为True的索引，即为峰值
    return np.where(P)[0]

# 示例调用
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sample_data import sample_data
    
    y = sample_data()
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(y)), y)
    px = ampd(y)
    plt.scatter(px, y[px], color="red")
    plt.show()
