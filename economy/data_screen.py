import os
import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|筛选有上升潜力的股票|')
parser.add_argument('--data_dir', default='dataset', type=str, help='|数据位置|')
parser.add_argument('--save_path', default='data_screen.txt', type=str, help='|筛选结果保存位置|')
parser.add_argument('--threshold', default=0.9, type=float, help='|筛选指标小于threshold的股票，1表示<=历史加权均值|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
def data_screen(data_dir, save_path, threshold):
    # 文件位置
    file_list = os.listdir(data_dir)
    screen_list = []
    for path in file_list:
        if '_add' not in path and '.csv' in path:  # 原始数据
            screen_list.append(path)
    file_list = screen_list
    path_list = [f'{data_dir}/{_}' for _ in file_list]
    # 筛选
    result_list = []
    close_metric_list = []
    for index, path in enumerate(path_list):
        df = pd.read_csv(path, index_col=0)
        # 收盘价筛选
        close_data = df['收盘价'].values
        ratio = 0.5 + np.arange(len(close_data)) / (len(close_data) - 1)
        mean = np.mean(close_data * ratio)
        close_metric = np.mean(close_data[-3:]) / mean
        if close_metric > threshold:
            continue
        # 记录
        result_list.append(f'{file_list[index].split(".")[0]} {close_metric:.2f}\n')
        close_metric_list.append(close_metric)
    result_list = sorted(result_list, key=lambda x: close_metric_list[result_list.index(x)])
    # 保存
    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(result_list)
    # 显示
    print(f'| 总数:{len(path_list)} | 筛选数:{len(result_list)} |')
    print(f'| 结果保存至:{save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    data_screen(args.data_dir, args.save_path, args.threshold)
