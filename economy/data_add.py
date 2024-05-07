import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|计算5日、10日、60日均线|')
parser.add_argument('--data_dir', default='dataset', type=str, help='|数据目录|')
parser.add_argument('--screen_path', default='data_screen.txt', type=str, help='|data_screen.py中筛选出的结果|')
parser.add_argument('--column', default='收盘价,成交量', type=str, help='|变量选择|')
args = parser.parse_args()
args.column = args.column.split(',')


# -------------------------------------------------------------------------------------------------------------------- #
def count(data, lengh, column):
    result_list = []
    for index in range(len(data) - lengh + 1):
        result_list.append(np.sum(data[index:index + lengh, :], axis=0) / lengh)
    result = np.stack(result_list, axis=0)
    column = [f'{_}_{lengh}' for _ in column]
    return result, column


def data_add(data_dir, screen_path, column):
    with open(screen_path, 'r', encoding='utf-8') as f:
        path_list = [f'{data_dir}/{_.split()[0]}.csv' for _ in f.readlines()]
        save_list = [f'{_[:-4]}_add.csv' for _ in path_list]
    for index, path in enumerate(path_list):
        df = pd.read_csv(path, index_col=0)
        value = df[column].values
        result_5, column_5 = count(data=value, lengh=5, column=column)
        result_10, column_10 = count(data=value, lengh=10, column=column)
        result_60, column_60 = count(data=value, lengh=60, column=column)
        lengh = len(result_60)
        result_5 = result_5[-lengh:]
        result_10 = result_10[-lengh:]
        result = np.concatenate([result_5, result_10, result_60], axis=1)
        column_add = column_5 + column_10 + column_60
        df_index = df.index[-lengh:]
        df_add = pd.DataFrame(result, columns=column_add, index=df_index)
        drop_index = df.index[:-lengh]
        df = df.drop(index=drop_index)
        df = pd.concat([df, df_add], axis=1)
        df.to_csv(save_list[index], header=True, index=True)
        print(f'| 结果保存至:{save_list[index]} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    data_add(args.data_dir, args.screen_path, args.column)
