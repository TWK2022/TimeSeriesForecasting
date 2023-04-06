import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='将excel的所有sheet合到一起')
parser.add_argument('--data_path', default=r'economy_m.xlsx', type=str, help='|数据路径|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    df_dict = pd.read_excel(args.data_path, sheet_name=None)
    key_all = tuple(df_dict.keys())
    column = df_dict[key_all[0]].columns
    data = []
    for key in key_all:
        data.append(df_dict[key][column].values)
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data, columns=column)
    df.to_csv(args.data_path.split('.xlsx')[0] + '.csv', index=False)
