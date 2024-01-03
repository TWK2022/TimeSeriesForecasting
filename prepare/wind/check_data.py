import argparse
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|删除所有含Nan的行|')
parser.add_argument('--data_path', default='dataset/浦发银行.csv', type=str, help='|数据保存的位置|')
parser.add_argument('--save_path', default='check.csv', type=str, help='|数据保存的位置|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    df = pd.read_csv(args.data_path)
    if df.isnull().values.any():
        input_ = input(f'>>>{args.data_path}中存在Nan(空值)，删除有Nan的行输入"1"，不删除输入其他<<<')
        if input_ == '1':
            df = df.dropna(axis=0, how='any')  # 删除所有有Nan的行
            df.to_csv(args.save_path)
            print(f'| 已删除所有含Nan的行，保存为{args.save_path} |')
