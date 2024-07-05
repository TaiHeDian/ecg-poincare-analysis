import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ampd import ampd
from baseline_correct import baseline_correct


path = "datas/jvp/"
filename = "gyx2.csv"
data_file = os.path.join(path, filename)

original_data = pd.read_csv(data_file, header=2)
jvp_data_times = original_data.iloc[:, [0, 2]].values

time = jvp_data_times[:, 0]
jvp_data = jvp_data_times[:, 1]

jvp_norm_data = baseline_correct(jvp_data, 500, 10, 6)
marked_points = ampd(jvp_norm_data)

plt.figure(figsize=(100,3))
plt.plot(jvp_norm_data, 'r')
plt.xlabel("Time (s)")  
plt.ylabel("ΔI/I₀ (%)")

plt.scatter(marked_points, 
            jvp_norm_data[marked_points],
            c="blue", 
            marker='x')

plt.savefig(f"{data_file.split('.')[0]}_marked.png")