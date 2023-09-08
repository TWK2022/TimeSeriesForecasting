import argparse
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|从WindPy获取最新股票数据以补全数据库|')
parser.add_argument('--data_path', default='dataset/浦发银行.csv', type=str, help='|数据保存的位置|')
parser.add_argument('--save_path', default='check.csv', type=str, help='|数据保存的位置|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    df = pd.read_csv(args.data_path)
    df = df.dropna(axis=0, how='any')
    df.to_csv(args.save_path)
