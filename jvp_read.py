import pandas as pd
import os

def read_data(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                # 使用第19行作为表头，从第21行开始读取数据
                data = pd.read_csv(file_path, header=14, skiprows=[15], sep=', ', engine='python')
                results[filename] = data
            except Exception as e:
                print(f"读取{filename}失败: {str(e)}")
    return results

if __name__ == '__main__':
    results = read_data('datas/jvp/commercial/')
    print(results.keys())
    