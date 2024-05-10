import yaml
import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|计算5日、10日、60日均线|')
parser.add_argument('--data_dir', default='dataset', type=str, help='|数据目录|')
parser.add_argument('--screen_path', default='data_screen.yaml', type=str, help='|data_screen.py中筛选出的结果|')
parser.add_argument('--column', default='收盘价,成交量', type=str, help='|选择计算的变量|')
parser.add_argument('--delete_column', default='市盈率(ttm),市净率,市销率(ttm)', type=str, help='|删除的变量|')
args = parser.parse_args()
args.column = args.column.split(',')
args.delete_column = args.delete_column.split(',')


# -------------------------------------------------------------------------------------------------------------------- #
def count(data, lengh, column):
    result_list = []
    for index in range(len(data) - lengh + 1):
        result_list.append(np.sum(data[index:index + lengh, :], axis=0) / lengh)
    result = np.stack(result_list, axis=0)
    column = [f'{_}_{lengh}' for _ in column]
    return result, column


def data_deal(args):
    with open(args.screen_path, 'r', encoding='utf-8') as f:
        yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
    for industry in yaml_dict:
        for name in yaml_dict[industry].keys():
            path = f'{args.data_dir}/{name}.csv'
            df = pd.read_csv(path, index_col=0)
            value = df[args.column].values
            result_5, column_5 = count(data=value, lengh=5, column=args.column)
            result_10, column_10 = count(data=value, lengh=10, column=args.column)
            result_60, column_60 = count(data=value, lengh=60, column=args.column)
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
            for delete in args.delete_column:
                if delete in df.columns:
                    df = df.drop(columns=delete)
            df.to_csv(f'{args.data_dir}/{name}_add.csv', header=True, index=True)
            print(f'| 结果保存至:{args.data_dir}/{name}_add.csv |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    data_deal(args)
