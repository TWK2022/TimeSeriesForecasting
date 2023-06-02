import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|将所有.csv合并，除了第一列的日期|')
parser.add_argument('--mark', default=True, type=bool, help='|为防止.csv中有重名，column前加上序号|')
parser.add_argument('--save_path', default=r'../dataset/z.csv', type=str, help='|保存路径|')
args = parser.parse_args()
args.data_dir = '../dataset/'
args.data_path = ['企业商品价格指数.csv', '其他存款性公司资产负债表.csv', '存款性公司概览.csv', '汇率.csv', '货币供应量.csv',
                  '货币当局资产负债表.csv', '金融机构人民币信贷收支表.csv', '金融机构外汇信贷收支表.csv', '金融机构本外币信贷收支表.csv']  # 合并的所有.csv路径
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    column_list = []
    data_list = []
    for i in range(len(args.data_path)):
        file_name = args.data_dir + args.data_path[i]
        df = pd.read_csv(file_name)
        if i == 0:
            column = [df.columns[0]]
            if args.mark:
                column.extend([f'{i}_' + _ for _ in df.columns[1:]])
            else:
                column.extend([_ for _ in df.columns[1:]])
            data = df[df.columns].values
        else:
            if args.mark:
                column = [f'{i}_' + _ for _ in df.columns[1:]]
            else:
                column = [_ for _ in df.columns[1:]]
            data = df[df.columns[1:]].values
        column_list.extend(column)
        data_list.append(data)
    data = np.concatenate(data_list, axis=1)
    df = pd.DataFrame(data, columns=column_list)
    df.to_csv(args.save_path, index=False)
    print(f'| column:{column_list} |')
    print(f'| 结果保存至:{args.save_path} |')
