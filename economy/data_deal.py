import os
import yaml
import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|计算5日、10日、20日均线|')
parser.add_argument('--data_dir', default='dataset', type=str, help='|数据目录|')
parser.add_argument('--number_path', default='tushare/number.yaml', type=str, help='|选择的股票|')
parser.add_argument('--column', default='收盘价,成交量', type=str, help='|选择计算的变量|')
args = parser.parse_args()
args.column = args.column.split(',')


# -------------------------------------------------------------------------------------------------------------------- #
def count(data, lengh, column):  # 计算均值
    result_list = []
    for index in range(len(data) - lengh + 1):
        result_list.append(np.mean(data[index:index + lengh, :], axis=0))
    result = np.stack(result_list, axis=0)
    column = [f'{_}_{lengh}' for _ in column]
    return result, column


def fix(data):  # 修复数据中间的单个nan值
    judge_list = np.isnan(data)
    if judge_list.any():  # 存在nan值
        index_list = np.arange(1, len(data) - 1)[judge_list[1:-1]]
        for index in index_list:
            data[index] = (data[index - 1] + data[index + 1]) / 2
    return data


def add_zero(data):  # 补充0值
    judge_list = np.isnan(data)
    if judge_list.any():  # 存在nan值
        index_list = np.arange(len(data))[judge_list]
        data[index_list] = 0
    return data


def reciprocal(data, numerator):  # 求导数
    judge_list = np.where(data == 0, False, True)
    index_list = np.arange(len(data))[judge_list]
    data[index_list] = numerator / data[index_list]
    return data


def data_deal(args):
    with open(args.number_path, 'r', encoding='utf-8') as f:
        yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
    for industry in yaml_dict:
        for name in yaml_dict[industry].keys():
            path = f'{args.data_dir}/{name}.csv'
            if not os.path.exists(path):
                print(f'| 文件不存在:{path} |')
                continue
            df = pd.read_csv(path, index_col=0)
            value = df[args.column].values
            # 数据太少舍弃
            if len(value) < 200:
                print(f'| 数据太少舍弃:{name} |')
                continue
            # 计算均线
            result_5, column_5 = count(data=value, lengh=5, column=args.column)
            result_10, column_10 = count(data=value, lengh=10, column=args.column)
            result_20, column_20 = count(data=value, lengh=20, column=args.column)
            lengh = len(result_20)
            result_5 = result_5[-lengh:]
            result_10 = result_10[-lengh:]
            result = np.concatenate([result_5, result_10, result_20], axis=1)
            column_add = column_5 + column_10 + column_20
            df_index = df.index[-lengh:]
            df_add = pd.DataFrame(result, columns=column_add, index=df_index)
            drop_index = df.index[:-lengh]
            df = df.drop(index=drop_index)
            df = pd.concat([df, df_add], axis=1)
            # 补充数据
            df['量比'] = fix(df['量比'].values)
            df['换手率(自由流通股)'] = fix(df['换手率(自由流通股)'].values)
            df['市盈率ttm'] = add_zero(df['市盈率ttm'].values)
            df['市净率'] = add_zero(df['市净率'].values)
            df['市销率ttm'] = add_zero(df['市销率ttm'].values)
            # 修改数据(由于亏损为0，原来值越小越好，现在改为值越大越好)
            df['市盈率ttm'] = reciprocal(df['市盈率ttm'].values, 100)
            df['市净率'] = reciprocal(df['市净率'].values, 1)
            df['市销率ttm'] = reciprocal(df['市销率ttm'].values, 1)
            df.rename(columns={'市盈率ttm': 'r市盈率ttm'}, inplace=True)
            df.rename(columns={'市净率': 'r市净率'}, inplace=True)
            df.rename(columns={'市销率ttm': 'r市销率ttm'}, inplace=True)
            # 保存
            df.to_csv(f'{args.data_dir}/{name}_add.csv', header=True, index=True)
            print(f'| 结果保存至:{args.data_dir}/{name}_add.csv |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    data_deal(args)
