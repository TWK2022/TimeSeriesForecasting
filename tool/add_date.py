import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|添加时间指标|')
parser.add_argument('--data_path', default=r'../dataset/merge.csv', type=str, help='|数据路径|')
parser.add_argument('--save_path', default=r'../dataset/all.csv', type=str, help='|保存路径|')
args = parser.parse_args()

# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    df = pd.read_csv(args.data_path)
    date = df[df.columns[0]].values
    data = df[df.columns[1:]].values
    DatetimeIndex = pd.DatetimeIndex(date)
    date = date[:, np.newaxis]
    month = DatetimeIndex.month[:, np.newaxis]
    dayofmonth = DatetimeIndex.day[:, np.newaxis]
    dayofweek = DatetimeIndex.dayofweek[:, np.newaxis]
    data = np.concatenate([date, month, dayofmonth, dayofweek, data], axis=1)
    column = ['date', 'month', 'dayofmonth', 'dayofweek']
    column.extend(df.columns[1:])
    df = pd.DataFrame(data, columns=column)
    df.to_csv(args.save_path, index=False)
    print(f'| 结果保存至:{args.save_path} |')
