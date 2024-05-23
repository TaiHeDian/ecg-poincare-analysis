import numpy as np
import matplotlib.pyplot as plt

def sample_data():
    N = 1000
    x = np.linspace(0, 200, N)
    y = 2 * np.cos(2 * np.pi * 300 * x) \
        + 5 * np.sin(2 * np.pi * 100 * x) \
        + 4 * np.random.randn(N)
    return y
