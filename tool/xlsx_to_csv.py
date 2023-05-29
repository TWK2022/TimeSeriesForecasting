import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|将excel的sheet合到一起，并转为.csv|')
parser.add_argument('--data_path', default=r'../dataset/金融机构本外币信贷收支表.xlsx', type=str, help='|数据路径|')
args = parser.parse_args()
args.sheet_name = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
                   '2020', '2021', '2022']  # args.sheet_name为读取sheet名称，填None读取所有sheet
args.column = ['month', '各项存款', '对国际金融机构负债', '各项贷款', '在国际金融机构资产', '资金运用总计']  # 取出的数据列名
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    df_dict = pd.read_excel(args.data_path, sheet_name=args.sheet_name)
    key_all = tuple(df_dict.keys())
    data = []
    for key in key_all:
        data.append(df_dict[key][args.column].values)
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data, columns=args.column)
    save_path = args.data_path.split('.xlsx')[0] + '.csv'
    df.to_csv(save_path, index=False)
    print(f'| sheet_name:{args.sheet_name} |')
    print(f'| column:{args.column} |')
    print(f'| 结果保存至:{save_path} |')
