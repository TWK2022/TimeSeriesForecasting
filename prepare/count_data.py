import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|计算5日、10日、60日均线|')
parser.add_argument('--data_path', default='dataset/华银电力.csv', type=str, help='|数据位置|')
parser.add_argument('--column', default='收盘价,成交量', type=str, help='|变量选择|')
args = parser.parse_args()
args.column = args.column.split(',')
args.save_path = args.data_path.split('.')[0] + '_add.csv'


# -------------------------------------------------------------------------------------------------------------------- #
def count(data, lengh, column):
    result_list = []
    for index in range(len(data) - lengh + 1):
        result_list.append(np.sum(data[index:index + lengh, :], axis=0) / lengh)
    result = np.stack(result_list, axis=0)
    column = [f'{_}_{lengh}' for _ in column]
    return result, column


def count_data():
    df = pd.read_csv(args.data_path, index_col=0)
    value = df[args.column].values
    result_5, column_5 = count(data=value, lengh=5, column=args.column)
    result_10, column_10 = count(data=value, lengh=10, column=args.column)
    result_60, column_60 = count(data=value, lengh=60, column=args.column)
    lengh = len(result_60)
    result_5 = result_5[-lengh:]
    result_10 = result_10[-lengh:]
    result = np.concatenate([result_5, result_10, result_60], axis=1)
    column = column_5 + column_10 + column_60
    index = df.index[-lengh:]
    df_add = pd.DataFrame(result, columns=column, index=index)
    drop_index = df.index[:-lengh]
    df = df.drop(index=drop_index)
    df = pd.concat([df, df_add], axis=1)
    df.to_csv(args.save_path, header=True, index=True)
    print(f'| 结果保存至:{args.save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    count_data()
